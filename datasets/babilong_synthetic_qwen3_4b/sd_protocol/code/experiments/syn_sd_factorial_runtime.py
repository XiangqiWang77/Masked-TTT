"""Frozen Qwen Teacher transport and native-Cartridges engine for the 2x2 study."""
import gc
import time


def load_engine(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cartridges.cache import AttnConfig
    from cartridges.models import FlexQwen3ForCausalLM
    from contextualized.engine import CartridgeEngine
    from experiments.flex_kernel_compat import install_generation_small_tiles
    engine = object.__new__(CartridgeEngine)
    engine.device = torch.device('cuda')
    engine.prefix_tokens = config['prefix_tokens']; engine.packed_tokens = config['packed_tokens']
    engine.generation_kernel_options = install_generation_small_tiles()
    engine.tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    engine.teacher = None
    if config.get('teacher_backend') != 'vllm_subprocess':
        engine.teacher = AutoModelForCausalLM.from_pretrained(config['model'],
            torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
            local_files_only=True).eval().requires_grad_(False)
    engine.student = FlexQwen3ForCausalLM.from_pretrained(config['model'],
        torch_dtype=torch.bfloat16, local_files_only=True).eval().requires_grad_(False)
    eos = engine.student.generation_config.eos_token_id
    engine.eos_ids = set(eos if isinstance(eos, list) else [eos]) | {engine.tokenizer.eos_token_id}
    engine.eos_ids.discard(None)
    c = engine.student.config
    engine.attn = AttnConfig(c.num_hidden_layers, c.num_key_value_heads, c.head_dim)
    return engine


def resident(engine, role):
    import torch
    other = engine.student if role == 'teacher' else engine.teacher
    if other is not None: other.to('cpu')
    gc.collect(); torch.cuda.empty_cache()
    (engine.teacher if role == 'teacher' else engine.student).to(engine.device)


class PrefixDecoder:
    """Reuse a frozen source prefix, cropping every question/candidate suffix.

    Input token IDs and normal Hugging Face attention are unchanged. The cache
    never contains Student memory or any preceding question's suffix.
    """
    def __init__(self, engine, prefix_ids, *, prefill_tokens, deadline):
        self.engine = engine; self.prefix_ids = list(prefix_ids)
        self.prefill_tokens = prefill_tokens; self.deadline = deadline; self.past = None

    def _check_time(self):
        if time.time() >= self.deadline:
            raise TimeoutError('Fixed factorial experiment deadline reached')

    def _forward(self, ids, past):
        import torch
        output = None
        for start in range(0, len(ids), self.prefill_tokens):
            self._check_time()
            tokens = torch.tensor([ids[start:start+self.prefill_tokens]],
                                  dtype=torch.long, device=self.engine.device)
            output = self.engine.teacher(input_ids=tokens, past_key_values=past,
                                         use_cache=True, logits_to_keep=1)
            past = output.past_key_values
        return output, past

    def generate(self, prompt_ids, *, max_new_tokens, temperature=0., topk=None):
        import torch
        if prompt_ids[:len(self.prefix_ids)] != self.prefix_ids or len(prompt_ids) <= len(self.prefix_ids):
            raise ValueError('Frozen source prefix differs from the current full prompt')
        maximum = self.engine.teacher.config.max_position_embeddings
        if len(prompt_ids)+max_new_tokens > maximum:
            raise ValueError('Full Teacher request exceeds model context capacity')
        generated, teacher_ids, teacher_logprobs = [], [], []
        with torch.no_grad():
            if self.past is None:
                _, self.past = self._forward(self.prefix_ids, None)
            self.past.crop(len(self.prefix_ids))
            try:
                output, self.past = self._forward(prompt_ids[len(self.prefix_ids):], self.past)
                for step in range(max_new_tokens):
                    self._check_time()
                    logits = output.logits[0, -1].float()
                    if topk is not None:
                        lp, indices = logits.log_softmax(-1).topk(topk)
                        teacher_ids.append(indices.detach().cpu())
                        teacher_logprobs.append(lp.detach().cpu())
                    token = (int(torch.multinomial((logits/temperature).softmax(-1), 1))
                             if temperature > 0 else int(logits.argmax()))
                    generated.append(token)
                    if token in self.engine.eos_ids: break
                    if step+1 < max_new_tokens:
                        output, self.past = self._forward([token], self.past)
            finally:
                self.past.crop(len(self.prefix_ids))
        return generated, teacher_ids, teacher_logprobs

    def clear(self):
        self.past = None


class FullDocumentTeacher:
    def __init__(self, engine, document, config, deadline):
        from contextualized.engine import SYSTEM_TEMPLATE
        self.engine = engine; self.document = document
        self.prefill_tokens = config['prefill_tokens']; self.deadline = deadline
        self.decoder = None; self.system = SYSTEM_TEMPLATE.format(subcorpus=document)

    def generate(self, question, *, max_new_tokens, temperature, topk):
        import torch
        from contextualized.engine import TrainExample
        ids = self.engine._chat_ids(question, self.document)
        if self.decoder is None:
            prefix = self.engine.tokenizer.apply_chat_template(
                [{'role': 'system', 'content': self.system}], tokenize=True,
                add_generation_prompt=False, enable_thinking=False)
            if ids[:len(prefix)] != prefix:
                raise ValueError('Teacher system prefix tokenization changed')
            self.decoder = PrefixDecoder(self.engine, prefix,
                prefill_tokens=self.prefill_tokens, deadline=self.deadline)
        tokens, indices, probabilities = self.decoder.generate(ids,
            max_new_tokens=max_new_tokens, temperature=temperature, topk=topk)
        return TrainExample(question, tokens, torch.stack(indices), torch.stack(probabilities),
                            bool(tokens and tokens[-1] in self.engine.eos_ids))

    def clear(self):
        if self.decoder is not None: self.decoder.clear()
        self.decoder = None


class SourceAssessmentGenerator:
    """Original single-user-message critics with one reusable source-only prefix."""
    def __init__(self, engine, document, config, deadline):
        from contextualized.engine import SYSTEM_TEMPLATE
        from experiments.selective_learning_runtime import FrozenAssessmentGenerator
        self.engine = engine; self.source_content = SYSTEM_TEMPLATE.format(subcorpus=document)
        self.config = config; self.deadline = deadline; self.decoder = None
        self.fresh = FrozenAssessmentGenerator(engine, prefill_tokens=config['prefill_tokens'])
        self.cached_calls = 0; self.fresh_calls = 0

    def __call__(self, content, max_new_tokens):
        if time.time() >= self.deadline: raise TimeoutError('Assessment deadline reached')
        if not content.startswith(self.source_content):
            self.fresh_calls += 1
            return self.fresh(content, max_new_tokens)
        tokenizer = self.engine.tokenizer
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if self.decoder is None:
            base = tokenizer.apply_chat_template([{'role': 'user', 'content': self.source_content}],
                tokenize=True, add_generation_prompt=False, enable_thinking=False)
            common = 0
            for left, right in zip(base, ids):
                if left != right: break
                common += 1
            prefix = ids[:max(1, common-4)]
            self.decoder = PrefixDecoder(self.engine, prefix,
                prefill_tokens=self.config['prefill_tokens'], deadline=self.deadline)
        if ids[:len(self.decoder.prefix_ids)] != self.decoder.prefix_ids:
            self.fresh_calls += 1
            return self.fresh(content, max_new_tokens)
        tokens, _, _ = self.decoder.generate(ids, max_new_tokens=max_new_tokens)
        self.cached_calls += 1
        complete = bool(tokens and tokens[-1] in self.engine.eos_ids)
        return {'prediction': tokenizer.decode(tokens, skip_special_tokens=True),
            'generated_token_ids': tokens, 'prompt_tokens': len(ids),
            'complete': complete, 'finish_reason': 'eos' if complete else 'length'}

    def clear(self):
        if self.decoder is not None: self.decoder.clear()
        self.decoder = None
