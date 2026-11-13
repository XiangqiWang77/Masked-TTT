"""Native-HF engine for the ratio Cartridge pipeline (OLMo-3; any verified native backbone).

Drop-in for experiments.syn_sd_factorial_runtime.load_engine, which hardwires the
Qwen3 FlexAttention student. Everything the ratio pipeline touches is provided:
``student`` (upstream packed interface), ``tokenizer``, ``attn``, ``eos_ids``,
``message_converter`` (native chat framing of saved Teacher targets), ``_chat_ids``,
``generate_student`` and a document initializer with the same contract as the
Qwen chunked KVFromText initializer (system-message framing, exact prefix length,
first token frozen). The frozen backbone also serves as the Dual-View scorer, so
no second copy is loaded and nothing is moved to CPU.
"""
from pathlib import Path

import torch


class NativeRatioEngine:
    native = True
    teacher = None
    training_kernel_options = None
    generation_kernel_options = None

    def __init__(self, config):
        from transformers import AutoTokenizer
        from cartridges.cache import AttnConfig
        from contextualized.native_backbone import (NativeCartridgeModel, load_native_model,
                                                    messages_to_element)
        self.device = torch.device('cuda')
        self.prefix_tokens = config.get('prefix_tokens', 512)
        self.packed_tokens = config.get('packed_tokens', 512)
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
        self.backbone = load_native_model(config['model'], self.device)
        self.student = NativeCartridgeModel(self.backbone)
        self.dual_teacher = self.backbone
        self.dual_teacher_resident = True
        self.message_converter = staticmethod(messages_to_element).__func__
        eos = self.backbone.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}
        self.eos_ids.discard(None)
        c = self.backbone.config
        self.attn = AttnConfig(c.num_hidden_layers, c.num_key_value_heads,
                               getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads)

    def _chat_ids(self, question, context=None):
        from contextualized.native_backbone import cartridge_prompt_ids
        if context is not None:
            raise ValueError('The ratio pipeline never evaluates a Cartridge with text context')
        return cartridge_prompt_ids(self.tokenizer, question)

    def system_prefix_ids(self, document, prefix_tokens):
        """Native equivalent of qwen_tokenize_data_into_system_prompt."""
        from contextualized.native_engine import native_system_closing_ids
        tokens = self.tokenizer.apply_chat_template([{'role': 'system', 'content': document}],
            tokenize=True, add_generation_prompt=False)
        end = native_system_closing_ids(self.tokenizer)
        if len(tokens) < prefix_tokens:
            raise ValueError('Source is shorter than the requested native KV prefix')
        if len(tokens) > prefix_tokens:
            tokens = tokens[:prefix_tokens - len(end)] + end
        assert len(tokens) == prefix_tokens
        return tokens

    def initialize_document(self, document, prefix_tokens, *, chunk_tokens=None, deadline=float('inf')):
        import time
        from contextualized.native_backbone import initialize_from_ids
        if time.time() >= deadline:
            raise TimeoutError('Native KV initialization reached the job deadline')
        ids = torch.tensor([self.system_prefix_ids(document, prefix_tokens)], device=self.device)
        return initialize_from_ids(self.backbone, ids, num_frozen_tokens=1)

    @torch.no_grad()
    def generate_student(self, question, cache, *, max_new_tokens=192):
        from contextualized.native_backbone import greedy_generate
        ids = torch.tensor([self._chat_ids(question)], device=self.device)
        tokens = greedy_generate(self.backbone, ids, cache, eos_ids=self.eos_ids,
                                 max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), tokens


def is_native(config):
    model_type = __import__('json').loads((Path(config['model']) / 'config.json').read_text())['model_type']
    return model_type != 'qwen3'
