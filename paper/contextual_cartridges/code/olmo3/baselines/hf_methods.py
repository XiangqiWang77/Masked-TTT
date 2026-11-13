"""Frozen Qwen adapters. Attention Matching calls the pinned authors' code."""
import gc
import json
import math
from pathlib import Path
import time

import torch
from transformers import DynamicCache

from baselines.protocol import bounded_context_ids, cache_statistics
from experiments.run_figure4_contextual import normalize_answer, longhealth_score, chr_f


class FrozenBaselines:
    def __init__(self, config, deadline):
        from evaluation.utils import load_model_and_tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(config['model'], device='cuda')
        self.model.requires_grad_(False)
        self.config, self.deadline = config, deadline
        self.device = next(self.model.parameters()).device
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.eos.add(self.tokenizer.eos_token_id)

    def check_time(self):
        if time.time() >= self.deadline:
            raise TimeoutError('Six-hour baseline allocation deadline')

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def question_ids(self, prompt):
        return self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)

    def format_document(self, document):
        from evaluation.utils import format_context
        return format_context(self.tokenizer, document, model_name=self.config['model'])

    @torch.no_grad()
    def forward(self, ids, cache):
        output = None
        for start in range(0, len(ids), self.config['prefill_chunk_tokens']):
            self.check_time()
            token_ids = torch.tensor([ids[start:start+self.config['prefill_chunk_tokens']]],
                                     device=self.device, dtype=torch.long)
            output = self.model(input_ids=token_ids, past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
        if output is None:
            raise ValueError('Cannot forward an empty prompt')
        return output, cache

    @torch.no_grad()
    def decode(self, ids, cache, max_tokens=192, sample=False):
        output, cache = self.forward(ids, cache)
        return self.decode_from_logits(output.logits[0,-1], cache, max_tokens, sample)

    @torch.no_grad()
    def decode_from_logits(self, logits, cache, max_tokens=192, sample=False):
        tokens = []
        for step in range(max_tokens):
            self.check_time()
            logits = logits.float()
            token = (int(torch.multinomial((logits / .7).softmax(-1), 1))
                     if sample else int(logits.argmax()))
            tokens.append(token)
            if token in self.eos:
                break
            if step + 1 < max_tokens:
                output, cache = self.forward([token], cache)
                logits = output.logits[0,-1]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def make_source_cache(self, document, prompts):
        # Use the native model window, not an artificially small ICL budget.
        # Reserve the same answer budget plus the longest evaluated question.
        maximum = int(self.model.config.max_position_embeddings)
        reserve = max(len(self.question_ids(p)) for p in prompts) + 1024
        raw = self.ids(document)
        wrapper = len(self.ids(self.format_document('')))
        retained, truncated = bounded_context_ids(raw, maximum - reserve - wrapper - 16)
        text = self.tokenizer.decode(retained, skip_special_tokens=False)
        context_ids = self.ids(self.format_document(text))
        if len(context_ids) + reserve > maximum:
            raise ValueError('Retokenized ICL prefix exceeds native model window')
        # Long source KV lives on CPU and only the current layer moves to GPU.
        offload = len(context_ids) > self.config['gpu_resident_source_tokens']
        if self.sliding_model() and not offload:
            # OLMo-3: keep every slot on every layer (sliding masks still apply),
            # so SnapKV sees all keys and crop() after the question is exact.
            cache = DynamicCache()
        else:
            cache = DynamicCache(config=self.model.config, offloading=offload,
                                 offload_only_non_sliding=False)
        _, cache = self.forward(context_ids, cache)
        stats = {'document_tokens': len(raw), 'retained_document_tokens': len(retained),
                 'truncated': truncated, 'truncation_policy': 'native_window_head_tail',
                 'native_window_tokens': maximum, 'cpu_offloaded': offload,
                 'prefix_tokens': len(context_ids), 'full_document': not truncated}
        return cache, stats

    def sliding_model(self):
        return 'sliding_attention' in (getattr(self.model.config, 'layer_types', None) or [])

    def native_sliding_candidates(self):
        """Recent source slots a sliding layer may keep: window minus question/answer reserve."""
        if not self.sliding_model():
            return 0
        return int(self.model.config.sliding_window) - 1024

    @torch.no_grad()
    def evaluate(self, cache, prefix_length, case, dataset):
        prompt = self.question_ids(case['prompt'])
        if dataset == 'qasper':
            answer = self.ids(f"<answer>\n{case['reference']}\n</answer>")
            raw = prompt + answer
            positions = torch.arange(len(prompt)-1, len(prompt)-1+len(answer),
                                     device=self.device)
            result = self.model(input_ids=torch.tensor([raw], device=self.device),
                past_key_values=cache, use_cache=True, logits_to_keep=positions)
            score = torch.nn.functional.cross_entropy(result.logits[0].float(),
                                                      torch.tensor(answer, device=self.device))
            return None, float(score)
        prediction = self.decode(prompt, cache, max_tokens=192)
        if dataset == 'mtob_ke':
            prediction = normalize_answer(prediction)
        score = longhealth_score(prediction, case) if dataset == 'longhealth' else chr_f(prediction, case['reference'])
        return prediction, score

    @torch.no_grad()
    def evaluate_compacted(self, compacted, original_length, case, dataset):
        from models.cache import CompactedPrefixCache
        cache = CompactedPrefixCache(compacted, original_seq_len=original_length, clone=False)
        return self.evaluate(cache, original_length, case, dataset)

    @torch.no_grad()
    def evaluate_snap(self, compacted, queries, original_length, case, dataset):
        """Continue the observed question; do not discard and recompute its KV."""
        from models.cache import CompactedPrefixCache
        retained=[]
        for index,(k,b,v) in enumerate(compacted):
            _,qk,qv=queries[index]
            qk,qv=qk.to(self.device),qv.to(self.device)
            retained.append((torch.cat((k,qk),dim=2),
                             torch.cat((b,torch.zeros(qk.shape[:-1],device=b.device,dtype=b.dtype)),dim=2),
                             torch.cat((v,qv),dim=2)))
        cache=CompactedPrefixCache(tuple(retained),
            original_seq_len=original_length+self.snap_question_length,clone=False)
        first=self.snap_initial_logits.to(self.device)
        if dataset=='qasper':
            answer=self.ids(f"<answer>\n{case['reference']}\n</answer>")
            scores=[first[None]]
            if len(answer)>1:
                output=self.model(input_ids=torch.tensor([answer[:-1]],device=self.device),
                    past_key_values=cache,use_cache=True,logits_to_keep=0)
                scores.append(output.logits[0])
            loss=torch.nn.functional.cross_entropy(torch.cat(scores).float(),
                                                  torch.tensor(answer,device=self.device))
            return None,float(loss)
        prediction=self.decode_from_logits(first,cache,max_tokens=192)
        if dataset=='mtob_ke':
            prediction=normalize_answer(prediction)
        score=longhealth_score(prediction,case) if dataset=='longhealth' else chr_f(prediction,case['reference'])
        return prediction,score

    def source_bytes(self, cache):
        return sum(t.numel()*t.element_size() for layer in cache.layers
                   for t in (layer.keys, layer.values))

    def summarize(self, document, budget, intermediate_path):
        """Generic hierarchical summary; no real question or answer is passed in."""
        ids = self.ids(document)
        intermediate_path = Path(intermediate_path)
        if len(ids) > self.config['summary_chunk_tokens']:
            if intermediate_path.exists():
                text = json.loads(intermediate_path.read_text())['text']
            else:
                summaries = []
                for start in range(0, len(ids), self.config['summary_chunk_tokens']):
                    chunk = self.tokenizer.decode(ids[start:start+self.config['summary_chunk_tokens']])
                    summaries.append(self._summary(chunk, 256))
                text = '\n\n'.join(summaries)
                # Hierarchical reduction keeps every chunk represented.
                while len(self.ids(text)) > self.config['summary_chunk_tokens']:
                    old_ids = self.ids(text)
                    summaries = [self._summary(self.tokenizer.decode(
                        old_ids[i:i+self.config['summary_chunk_tokens']]), 256)
                        for i in range(0, len(old_ids), self.config['summary_chunk_tokens'])]
                    text = '\n\n'.join(summaries)
                from experiments.run_figure4_contextual import write
                write(intermediate_path, {'text': text, 'original_tokens': len(ids),
                                         'query_used': False, 'gold_used': False})
        else:
            text = document
        return self._summary(text, budget)

    def _summary(self, text, budget):
        prompt = (f'Summarize the following text, preserving key facts, definitions, '
                  f'conditions and relations, in at most {budget} tokens.\n\n{text}\n\nSummary:')
        result = self.decode(self.question_ids(prompt),
            DynamicCache(config=self.model.config), max_tokens=budget)
        # Generation token budget is explicit; do not claim every summary fills it.
        return result

    @torch.no_grad()
    def attention_matching(self, document, budget, *, include_wrapper_in_budget=False):
        from evaluation.configs.utils import load_algorithm_config, load_query_config
        from evaluation.utils import compute_article_indices
        from compaction.compaction_methods.registry import get_compaction_method
        # The paper's named fast variant: on-policy AM-HighestAttnKeys + repeat.
        kwargs = dict(load_algorithm_config('default')['AM-HighestAttnKeys'])
        kwargs.update(precomputed_budget_path=self.config['head_budget_path'],
                      chunking='fixed', chunk_size=self.config['am_chunk_tokens'],
                      use_kv_based=False)
        method = get_compaction_method('AM-HighestAttnKeys-fast', kwargs)
        formatted = self.format_document(document)
        indices = compute_article_indices(self.tokenizer, formatted, document)
        wrapper = len(self.ids(formatted)) - len(indices)
        # The released text-chunked wrapper iterates this argument, including
        # for full-attention Qwen models. None is not an empty set of layers.
        sliding_layers = {i for i, kind in enumerate(
            getattr(self.model.config, 'layer_types', None) or [])
            if kind == 'sliding_attention'}
        # OLMo-3: AM (unchanged) runs with every layer treated as global; its 24
        # sliding layers are then replaced by native last-window raw keys below.
        olmo_sliding = set(sliding_layers) if self.model.config.model_type == 'olmo3' else set()
        if olmo_sliding:
            sliding_layers = set()
        compacted, stats = method.compact_kv_cache(
            past_key_values=None, target_size=budget if include_wrapper_in_budget else budget+wrapper, indices=indices,
            query_config=load_query_config('repeat'), model=self.model,
            tokenizer=self.tokenizer, formatted_context=formatted,
            compute_stats=False, article_text=document,
            sliding_layer_indices=sliding_layers)
        self.check_time()
        if olmo_sliding:
            # Sliding layers never attend beyond their window: keep the most recent
            # raw K/V of the same formatted context (same per-layer slot budget as
            # the compacted layers, and inside the window for question + answer).
            cap = min(budget if include_wrapper_in_budget else budget + wrapper, self.native_sliding_candidates())
            source = DynamicCache()
            _, source = self.forward(self.ids(formatted), source)
            replaced = list(compacted)
            for i in olmo_sliding:
                k, v = source.layers[i].keys[:, :, -cap:], source.layers[i].values[:, :, -cap:]
                ref = compacted[i][0]
                k, v = k.to(ref.device, ref.dtype).contiguous(), v.to(ref.device, ref.dtype).contiguous()
                replaced[i] = (k, torch.zeros(k.shape[:-1], dtype=k.dtype, device=k.device), v)
            compacted = tuple(replaced)
            del source
            stats['sliding_layer_policy'] = (f'OLMo-3: AM on the {len(compacted)-len(olmo_sliding)} full-attention layers; '
                                             f'{len(olmo_sliding)} sliding layers keep the most recent {cap} raw K/V')
        # Chunked wrapper performs its own RoPE correction and records the
        # reconstructed logical sequence length; never use physical padded size.
        stats['original_sequence_length'] = len(self.ids(formatted))
        stats['variant'] = 'AM-HighestAttnKeys-fast, authors text-chunked wrapper'
        return compacted, stats

    @torch.no_grad()
    def snap_queries(self, source_cache, source_length, prompt):
        """Capture Qwen's post-normalization, post-RoPE question queries."""
        from models.qwen3.modeling_qwen3 import rotate_half
        queries = {}
        hooks = []
        window = self.config['snap_window_tokens']
        def hook(index):
            def collect(module, args, kwargs):
                hidden = args[0] if args else kwargs['hidden_states']
                hidden = hidden[:, -window:]
                if self.model.config.model_type == 'olmo3':
                    # OLMo-3 normalizes the whole projection before splitting heads.
                    q = module.q_norm(module.q_proj(hidden)).view(1, hidden.shape[1], -1, module.head_dim)
                else:
                    q = module.q_norm(module.q_proj(hidden).view(1, hidden.shape[1], -1, module.head_dim))
                q = q.transpose(1, 2)
                cos, sin = kwargs['position_embeddings']
                cos, sin = cos[:, -hidden.shape[1]:].unsqueeze(1), sin[:, -hidden.shape[1]:].unsqueeze(1)
                q = (q*cos + rotate_half(q)*sin).to(hidden.dtype)  # OLMo-3 RoPE runs in fp32
                prior = queries.get(index)
                q = q.cpu()
                queries[index] = (torch.cat((prior, q), dim=2) if prior is not None else q)[:, :, -window:]
            return collect
        try:
            for index, layer in enumerate(self.model.model.layers):
                hooks.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
            question_ids=self.question_ids(prompt)
            output,_=self.forward(question_ids, source_cache)
            self.snap_initial_logits=output.logits[0,-1].cpu()
            self.snap_question_length=len(question_ids)
            # Preserve the full question's keys for the attention softmax
            # denominator, including the causal mask within its observation.
            for index, layer in enumerate(source_cache.layers):
                queries[index] = (queries[index], layer.keys[:, :, source_length:].cpu().clone(),
                                  layer.values[:, :, source_length:].cpu().clone())
        finally:
            for handle in hooks:
                handle.remove()
            source_cache.crop(source_length)
        return queries

    @torch.no_grad()
    def snap_cache(self, source_cache, queries, budget):
        """Content-only GQA SnapKV: actual-query observation, mean scores, avg pooling.

        The observed question KV and first decoding logits are kept unchanged.
        This is explicitly not the AM paper's query-agnostic mock-Q baseline.
        """
        compacted = []
        types = getattr(self.model.config, 'layer_types', None) or []
        window = self.native_sliding_candidates()
        for index, layer in enumerate(source_cache.layers):
            self.check_time()
            keys, values = layer.keys.to(self.device), layer.values.to(self.device)
            if window and index < len(types) and types[index] == 'sliding_attention':
                # OLMo-3 sliding layers never attend beyond their window; offer only
                # keys that stay inside it for the whole question + answer.
                keys, values = keys[:, :, -window:], values[:, :, -window:]
            q, question_keys = (t.to(self.device) for t in queries[index][:2])
            groups = q.shape[1] // keys.shape[1]
            kk, bb, vv = [], [], []
            for head in range(keys.shape[1]):
                head_q = q[0, head*groups:(head+1)*groups].reshape(-1, q.shape[-1])
                count = min(budget, keys.shape[-2])
                all_keys = torch.cat((keys[0, head], question_keys[0, head]), dim=0)
                logits = (head_q @ all_keys.T).float() / math.sqrt(keys.shape[-1])
                observed = q.shape[-2]
                question_length = question_keys.shape[-2]
                query_positions = torch.arange(question_length-observed, question_length,
                                                device=self.device).repeat(groups)
                future = torch.arange(question_length, device=self.device)[None] > query_positions[:, None]
                logits[:, keys.shape[-2]:].masked_fill_(future, -torch.inf)
                scores = logits.softmax(-1)[:, :keys.shape[-2]].mean(0)
                scores = torch.nn.functional.avg_pool1d(scores[None,None], kernel_size=5,
                                                        padding=2, stride=1)[0,0]
                chosen = scores.topk(count).indices.sort().values
                k, v = keys[0,head,chosen], values[0,head,chosen]
                b = torch.zeros(count, dtype=keys.dtype, device=self.device)
                kk.append(k); bb.append(b); vv.append(v)
                del logits, scores, all_keys
            compacted.append((torch.stack(kk)[None], torch.stack(bb)[None], torch.stack(vv)[None]))
            del keys, values, q, question_keys
        return tuple(compacted)
