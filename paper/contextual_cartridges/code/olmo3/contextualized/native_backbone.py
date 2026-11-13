"""Native HF backbone adapter for the original Cartridges K/V container.

OLMo-3 and GPT-OSS are not in Stanford's released Flex model implementations.
This adapter keeps their own attention, sliding windows, RoPE, and attention
sinks. It supplies only native K/V tensors; no forward hooks or new parameters
are introduced. Packed conversations are evaluated independently so they cannot
attend to one another. This extension requires parity and gradient tests before
benchmark use and is not part of the official Qwen reproduction.
"""
from types import SimpleNamespace

import numpy as np
import torch
from transformers import DynamicCache

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import DatasetElement, TokenCounts


def native_cache(cartridge):
    pairs = []
    for index in range(cartridge.config.n_layers):
        keys = [cartridge.trainable_keys[index]]
        values = [cartridge.trainable_values[index]]
        if cartridge._num_frozen_tokens:
            keys.insert(0, cartridge.frozen_keys[index])
            values.insert(0, cartridge.frozen_values[index])
        pairs.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    # Keep all memory slots in storage. Native per-layer sliding attention masks
    # still apply, so older slots are inaccessible to a sliding layer as usual.
    return DynamicCache(ddp_cache_data=pairs)


class NativeCartridgeModel(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config

    @property
    def device(self):
        return self.backbone.device

    def forward(self, input_ids, seq_ids, position_ids, past_key_values, use_cache=True, mode='train',
                logits_to_keep=None):
        """Upstream packed interface. ``mode='generate'`` is the same teacher-forced
        forward (the native cache is rebuilt per call, so no transient KV is ever
        staged into the Cartridge). ``logits_to_keep`` is a sorted 1-D tensor of
        packed positions; rows are returned in that order, as upstream does."""
        if mode not in ('train', 'generate'):
            raise ValueError(f'Unsupported native adapter mode {mode}')
        if input_ids.ndim != 1 or position_ids.ndim != 1:
            raise ValueError('Expected the upstream one-dimensional packed token layout')
        if logits_to_keep is not None and not (torch.is_tensor(logits_to_keep) and logits_to_keep.ndim == 1):
            raise ValueError('Native adapter expects logits_to_keep as a 1-D position tensor')
        if logits_to_keep is not None:
            return self._selected(input_ids, seq_ids, position_ids, past_key_values, logits_to_keep)
        prefix = past_key_values.num_cartridge_tokens()
        outputs, indices = [], []
        for sequence in torch.unique(seq_ids).tolist():
            if sequence < 0:
                continue
            index = torch.nonzero(seq_ids == sequence, as_tuple=False).flatten()
            # Upstream collate pads with token=0, sequence=0, position=0.
            # Those trailing padding slots are not part of conversation zero.
            # Passing them to native FA2 makes its packed-sequence detector
            # mistake the position reset for another conversation.
            local_positions = position_ids[index]
            resets = torch.nonzero(local_positions != torch.arange(len(index), device=index.device),
                                   as_tuple=False).flatten()
            if len(resets):
                stop = int(resets[0])
                if (sequence != 0 or stop == 0 or bool(local_positions[stop:].any())
                        or bool(input_ids[index[stop:]].any())):
                    raise ValueError('Unexpected noncontiguous native conversation positions')
                index = index[:stop]
            positions = position_ids[index] + prefix
            result = self.backbone(input_ids=input_ids[index][None],
                position_ids=positions[None], cache_position=positions,
                attention_mask=torch.ones((1, prefix+len(index)), device=input_ids.device, dtype=torch.long),
                past_key_values=native_cache(past_key_values), use_cache=True)
            outputs.append(result.logits[0])
            indices.append(index)
        if not outputs:
            raise ValueError('Packed batch contains no conversation tokens')
        logits = outputs[0].new_zeros((len(input_ids), self.config.vocab_size))
        logits = logits.index_copy(0, torch.cat(indices), torch.cat(outputs))
        return SimpleNamespace(logits=logits[None])

    def _sequences(self, input_ids, seq_ids, position_ids):
        """Same per-conversation split and padding rule as forward()."""
        for sequence in torch.unique(seq_ids).tolist():
            if sequence < 0:
                continue
            index = torch.nonzero(seq_ids == sequence, as_tuple=False).flatten()
            local_positions = position_ids[index]
            resets = torch.nonzero(local_positions != torch.arange(len(index), device=index.device),
                                   as_tuple=False).flatten()
            if len(resets):
                stop = int(resets[0])
                if (sequence != 0 or stop == 0 or bool(local_positions[stop:].any())
                        or bool(input_ids[index[stop:]].any())):
                    raise ValueError('Unexpected noncontiguous native conversation positions')
                index = index[:stop]
            yield index

    def _selected(self, input_ids, seq_ids, position_ids, past_key_values, keep):
        keep = keep.to(input_ids.device)
        if len(keep) > 1 and not bool((keep[1:] > keep[:-1]).all()):
            raise ValueError('logits_to_keep must be strictly increasing')
        prefix = past_key_values.num_cartridge_tokens()
        rows, where = [], []
        for index in self._sequences(input_ids, seq_ids, position_ids):
            wanted = torch.isin(index, keep)
            if not bool(wanted.any()):
                continue  # no supervised row in this conversation
            local = torch.nonzero(wanted, as_tuple=False).flatten()
            positions = position_ids[index] + prefix
            result = self.backbone(input_ids=input_ids[index][None],
                position_ids=positions[None], cache_position=positions,
                attention_mask=torch.ones((1, prefix+len(index)), device=input_ids.device, dtype=torch.long),
                past_key_values=native_cache(past_key_values), use_cache=True, logits_to_keep=local)
            rows.append(result.logits[0])
            where.append(torch.searchsorted(keep, index[local]))
        if not rows:
            raise ValueError('No requested logits row belongs to a conversation')
        where = torch.cat(where)
        if len(where) != len(keep):
            raise ValueError('Some requested logits rows are padding, not conversation tokens')
        logits = rows[0].new_empty((len(keep), self.config.vocab_size))
        logits = logits.index_copy(0, where, torch.cat(rows))
        return SimpleNamespace(logits=logits[None])


def cartridge_prompt_ids(tokenizer, question):
    """Render the native continuation after a system/document KV prefix.

    A user-only template may inject a default system message (OLMo and
    GPT-OSS both do). That scaffold is already part of the Cartridge prefix.
    Derive the user/assistant continuation from the template itself instead
    of inserting the default scaffold again or hardcoding model delimiters.
    """
    system = {'role': 'system', 'content': '__cartridge_template_boundary__'}
    prefix = tokenizer.apply_chat_template([system], tokenize=True,
        add_generation_prompt=False, enable_thinking=False)
    whole = tokenizer.apply_chat_template([system, {'role': 'user', 'content': question}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if whole[:len(prefix)] != prefix or len(whole) == len(prefix):
        raise ValueError('Native template does not expose a stable system-to-user boundary')
    return whole[len(prefix):]


def messages_to_element(messages, *, tokenizer):
    """Align original sparse Teacher targets to each backbone's native template."""
    if len(messages) != 2 or [m.role for m in messages] != ['user', 'assistant']:
        raise ValueError('The current upstream self-study recipe requires a single A/B round')
    answer = messages[1]
    prompt = cartridge_prompt_ids(tokenizer, messages[0].content)
    if answer.token_ids is None or answer.top_logprobs is None:
        raise ValueError('Teacher token IDs and distributions must be preserved')
    targets = answer.top_logprobs
    return DatasetElement(
        input_ids=torch.tensor(prompt+list(answer.token_ids), dtype=torch.long),
        topk_token_ids=torch.from_numpy(np.asarray(targets.token_id)),
        topk_logprobs=torch.from_numpy(np.asarray(targets.logprobs)),
        topk_token_idxs=torch.from_numpy(np.asarray(targets.token_idx)+len(prompt)),
        metadata=[], token_counts=TokenCounts(num_system_and_user_tokens=len(prompt),
                                             num_assistant_tokens=len(answer.token_ids)))


@torch.no_grad()
def initialize_from_ids(backbone, input_ids, *, num_frozen_tokens=0):
    """Initialize unmodified TrainableCache tensors from the native model cache."""
    config = backbone.config
    attn = AttnConfig(config.num_hidden_layers, config.num_key_value_heads,
                      getattr(config, 'head_dim', None) or config.hidden_size//config.num_attention_heads)
    output = backbone(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
    layers = output.past_key_values.layers
    if any(layer.keys.shape[-2] != input_ids.shape[-1] for layer in layers):
        raise ValueError('Prefix initialization unexpectedly cropped native memory slots')
    return TrainableCache(attn, init_keys=[layer.keys.detach().clone() for layer in layers],
                          init_values=[layer.values.detach().clone() for layer in layers],
                          num_frozen_tokens=num_frozen_tokens).to(input_ids.device)


@torch.no_grad()
def greedy_generate(backbone, input_ids, cartridge, *, eos_ids, max_new_tokens):
    past = native_cache(cartridge)
    current, generated = input_ids, []
    for _ in range(max_new_tokens):
        positions = torch.arange(past.get_seq_length(), past.get_seq_length()+current.shape[-1],
                                 device=current.device)
        output = backbone(input_ids=current, position_ids=positions[None], cache_position=positions,
                          past_key_values=past, use_cache=True, logits_to_keep=1)
        past = output.past_key_values
        token = output.logits[0, -1].argmax().item()
        if token in eos_ids:
            break
        generated.append(token)
        current = torch.tensor([[token]], device=input_ids.device)
    return generated


def load_native_model(path, device='cuda'):
    from transformers import AutoConfig, AutoModelForCausalLM, Mxfp4Config
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    kwargs = {'local_files_only': True, 'dtype': torch.bfloat16}
    if config.model_type == 'gpt_oss':
        kwargs.update(attn_implementation='flex_attention', quantization_config=Mxfp4Config(dequantize=True))
    elif config.model_type in {'olmo3', 'qwen3'}:
        kwargs.update(attn_implementation='flash_attention_2')
    else:
        raise ValueError(f'Native Cartridge adapter has not been verified for {config.model_type}')
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()
    model.requires_grad_(False)
    return model
