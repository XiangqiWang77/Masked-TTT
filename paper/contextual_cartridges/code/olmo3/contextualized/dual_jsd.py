"""Dual-View JSD self-distillation targets: evidence Teacher E and Whole-KV W.

For every saved Teacher trajectory (answer tokens are reused, never
regenerated) the same frozen model is teacher-forced twice:

  E: [trajectory's own Teacher context] + question + answer
     (specific arm: the bound evidence; agnostic arm: its synthesis chunk)
  W: [whole document, native-window head/tail as Whole KV] + question + answer

On the union of both top-k supports plus one rest bucket per view,

  M_t   = (P_E,t + P_W,t) / 2
  JSD_t = H(M_t) - (H(P_E,t) + H(P_W,t)) / 2     (bits, in [0, 1])
  w_t   = 1 - JSD_t

The stored sparse target is w_t * M_t on the retained support, so the
unchanged upstream sparse soft-target loss becomes w_t * CE(M_t, Student_t).
Sampler, optimizer, update budget and early-stopping rule stay those of SD.
No benchmark answer, judge, or Student signal enters the targets.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

VERSION = 'dual-view-jsd-v1'


def bucket_distributions(log_e, log_w, k):
    """Exact probabilities of both views on union(top-k E, top-k W) + rest.

    log_e, log_w: [T, V] natural-log full-vocabulary distributions.
    Returns ids [T, U] (-1 padded), p_e/p_w [T, U+1] with the rest bucket last.
    """
    top = torch.cat([log_e.topk(k, dim=-1).indices, log_w.topk(k, dim=-1).indices], dim=-1)
    sorted_ids, _ = top.sort(dim=-1)
    duplicate = torch.zeros_like(sorted_ids, dtype=torch.bool)
    duplicate[:, 1:] = sorted_ids[:, 1:] == sorted_ids[:, :-1]
    ids = sorted_ids.masked_fill(duplicate, -1)
    valid = ids >= 0
    gather = ids.clamp_min(0)
    p_e = log_e.gather(-1, gather).exp() * valid
    p_w = log_w.gather(-1, gather).exp() * valid
    rest_e = (1 - p_e.sum(-1, keepdim=True)).clamp_min(0)
    rest_w = (1 - p_w.sum(-1, keepdim=True)).clamp_min(0)
    return ids, torch.cat([p_e, rest_e], -1), torch.cat([p_w, rest_w], -1)


def entropy_bits(p):
    return -(p * torch.log2(p.clamp_min(1e-30))).sum(-1)


def dual_view_terms(p_e, p_w):
    m = (p_e + p_w) / 2
    jsd = (entropy_bits(m) - (entropy_bits(p_e) + entropy_bits(p_w)) / 2).clamp(0, 1)
    return m, jsd, 1 - jsd


def sparse_weighted_target(ids, m, weight, *, target_mass=.99, max_entries=40):
    """Rows of (token_idx, token_id, logprob) for w_t * M_t, highest M first.

    Entries are kept until they cover target_mass of M_t (as upstream
    flatten(.99) does for the single Teacher); the rest bucket is never a
    trainable token, exactly as in the stored single-Teacher targets.
    """
    support = m[:, :-1].masked_fill(ids < 0, -1.)
    order = support.argsort(dim=-1, descending=True)[:, :max_entries]
    probs = support.gather(-1, order)
    tokens = ids.gather(-1, order)
    before = probs.cumsum(-1) - probs
    keep = (probs > 0) & (tokens >= 0) & (before < target_mass)
    keep[:, 0] = tokens[:, 0] >= 0
    rows = torch.arange(len(ids), device=ids.device)[:, None].expand_as(tokens)
    logp = torch.log(probs.clamp_min(1e-30)) + torch.log(weight.clamp_min(1e-8))[:, None]
    return (rows[keep].cpu().numpy().astype(np.int64), tokens[keep].cpu().numpy().astype(np.int32),
            logp[keep].float().cpu().numpy().astype(np.float32))


def summarize(values, thresholds=(.1, .3, .5)):
    values = np.asarray(values, dtype=np.float64)
    out = {'mean': float(values.mean()), 'median': float(np.median(values)),
           'p90': float(np.quantile(values, .9)), 'max': float(values.max())}
    out.update({f'fraction_above_{t}': float((values > t).mean()) for t in thresholds})
    return out


class DualViewScorer:
    """Frozen HF Qwen teacher-forcing for both views; reuses one W prefix KV."""

    def __init__(self, model, tokenizer, *, prefill_tokens=2048, deadline=math.inf):
        self.model, self.tokenizer = model, tokenizer
        self.prefill_tokens, self.deadline = prefill_tokens, deadline
        self.device = next(model.parameters()).device

    def _check(self):
        if time.time() >= self.deadline:
            raise TimeoutError('Dual-view scoring reached the allocation deadline')

    def chat(self, system, question):
        return self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)

    def system_prefix(self, system):
        return self.tokenizer.apply_chat_template([{'role': 'system', 'content': system}],
            tokenize=True, add_generation_prompt=False, enable_thinking=False)

    @torch.no_grad()
    def prefill(self, ids):
        from transformers import DynamicCache
        # Sliding-window layers in a config-built cache drop old slots, which
        # breaks per-trajectory crop(). Keep every slot; the model's own
        # sliding masks still apply (OLMo-3). Qwen3 (no sliding) is unchanged.
        sliding = 'sliding_attention' in (getattr(self.model.config, 'layer_types', None) or [])
        cache = DynamicCache() if sliding else DynamicCache(config=self.model.config)
        for start in range(0, len(ids), self.prefill_tokens):
            self._check()
            chunk = torch.tensor([ids[start:start + self.prefill_tokens]], device=self.device)
            cache = self.model(input_ids=chunk, past_key_values=cache, use_cache=True,
                               logits_to_keep=1).past_key_values
        return cache

    @torch.no_grad()
    def answer_logprobs(self, prompt_ids, answer_ids, cache=None, offset=0):
        """[T, V] log-probabilities predicting each answer token (fp32)."""
        self._check()
        ids = list(prompt_ids[offset:]) + list(answer_ids)
        first = len(prompt_ids) - offset - 1
        positions = torch.arange(first, first + len(answer_ids), device=self.device)
        output = self.model(input_ids=torch.tensor([ids], device=self.device),
                            past_key_values=cache, use_cache=cache is not None,
                            logits_to_keep=positions)
        return output.logits[0].float().log_softmax(-1)


def whole_document_system(tokenizer, document, system_template, reserve_tokens, maximum):
    """Same deterministic head/tail retention rule as the Whole-KV baseline."""
    from baselines.protocol import bounded_context_ids
    raw = tokenizer.encode(document, add_special_tokens=False)
    wrapper = len(tokenizer.encode(system_template.format(subcorpus=''), add_special_tokens=False))
    retained, truncated = bounded_context_ids(raw, maximum - reserve_tokens - wrapper - 32)
    text = tokenizer.decode(retained, skip_special_tokens=False) if truncated else document
    return system_template.format(subcorpus=text), {
        'document_tokens': len(raw), 'retained_document_tokens': len(retained),
        'truncated': truncated, 'truncation_policy': 'native_window_head_tail',
        'reserved_continuation_tokens': reserve_tokens}


def build_dual_targets(scorer, arms, document, *, system_template, topk=20,
                       target_mass=.99, max_entries=40, min_e_agreement=.5):
    """arms: {name: [Conversation]} -> ({name: [Conversation]}, audit).

    One whole-document prefix KV serves every trajectory of every arm.
    """
    tokenizer = scorer.tokenizer
    maximum = int(scorer.model.config.max_position_embeddings)
    continuation = 0
    for conversations in arms.values():
        for c in conversations:
            user, assistant = c.messages
            if user.role != 'user' or assistant.role != 'assistant' or assistant.top_logprobs is None:
                raise ValueError('Expected one user turn and one Teacher-scored assistant turn')
            continuation = max(continuation, len(scorer.chat('', user.content)) + len(assistant.token_ids))
    whole_system, retention = whole_document_system(tokenizer, document, system_template,
                                                    continuation + 64, maximum)
    prefix = scorer.system_prefix(whole_system)
    started = time.monotonic()
    cache = scorer.prefill(prefix)
    retention['prefix_tokens'] = len(prefix)
    retention['prefill_seconds'] = time.monotonic() - started
    outputs, audit = {}, {'version': VERSION, 'whole_view': retention, 'topk_per_view': topk,
                          'target_mass': target_mass, 'max_entries': max_entries,
                          'weight': '1 - JSD_bits(P_E, P_W) on union top-k + rest',
                          'target': 'w_t * (P_E + P_W)/2', 'benchmark_answers_used': False,
                          'answer_tokens_regenerated': False, 'arms': {}}
    try:
        for arm, conversations in arms.items():
            rows, converted, all_jsd, all_w = [], [], [], []
            for index, c in enumerate(conversations):
                user, assistant = c.messages
                answer = list(assistant.token_ids)
                e_prompt = scorer.chat(c.system_prompt, user.content)
                log_e = scorer.answer_logprobs(e_prompt, answer)
                w_prompt = scorer.chat(whole_system, user.content)
                if w_prompt[:len(prefix)] != prefix:
                    raise ValueError('Whole-document system prefix tokenization changed')
                cache.crop(len(prefix))
                log_w = scorer.answer_logprobs(w_prompt, answer, cache=cache, offset=len(prefix))
                cache.crop(len(prefix))
                if log_e.shape != log_w.shape or log_e.shape[0] != assistant.top_logprobs.shape[0]:
                    raise ValueError('Teacher-forced positions do not match the saved trajectory')
                ids, p_e, p_w = bucket_distributions(log_e, log_w, topk)
                m, jsd, weight = dual_view_terms(p_e, p_w)
                token_idx, token_id, logprobs = sparse_weighted_target(
                    ids, m, weight, target_mass=target_mass, max_entries=max_entries)
                stored = assistant.top_logprobs
                first = np.zeros(stored.shape[0], dtype=np.int64) - 1
                seen = set()
                for r, t in zip(stored.token_idx.tolist(), stored.token_id.tolist()):
                    if r not in seen:
                        seen.add(r); first[r] = t
                e_top = log_e.argmax(-1).cpu().numpy()
                w_top = log_w.argmax(-1).cpu().numpy()
                answer_np = np.asarray(answer)
                new = copy.deepcopy(c)
                from cartridges.clients.base import FlatTopLogprobs
                new.messages[1].top_logprobs = FlatTopLogprobs(token_idx=token_idx, token_id=token_id,
                    logprobs=logprobs, shape=(stored.shape[0], max_entries))
                new.metadata = dict(new.metadata, dual_view_jsd=dict(version=VERSION,
                    mean_weight=float(weight.mean()), mean_jsd_bits=float(jsd.mean())))
                new.type = f'{c.type}+dual_view_jsd'
                converted.append(new)
                jsd_np, w_np = jsd.cpu().numpy(), weight.cpu().numpy()
                all_jsd.append(jsd_np); all_w.append(w_np)
                rows.append({'index': index, 'answer_tokens': len(answer),
                    'mean_weight': float(w_np.mean()), 'mean_jsd_bits': float(jsd_np.mean()),
                    'fraction_jsd_above_0.3': float((jsd_np > .3).mean()),
                    'e_w_top1_agreement': float((e_top == w_top).mean()),
                    'e_recompute_matches_saved_top1': float((e_top == first).mean()),
                    'answer_token_mean_logprob_e': float(log_e[torch.arange(len(answer)), answer].mean()),
                    'answer_token_mean_logprob_w': float(log_w[torch.arange(len(answer)), answer].mean()),
                    'answer_token_is_e_top1': float((e_top == answer_np).mean()),
                    'answer_token_is_w_top1': float((w_top == answer_np).mean()),
                    'target_entries': int(len(token_idx))})
                del log_e, log_w, ids, p_e, p_w, m
            jsd_cat, w_cat = np.concatenate(all_jsd), np.concatenate(all_w)
            agreement = float(np.mean([r['e_recompute_matches_saved_top1'] for r in rows]))
            audit['arms'][arm] = {'trajectories': rows, 'tokens': int(len(jsd_cat)),
                'jsd_bits': summarize(jsd_cat), 'weight': summarize(w_cat, (.5, .7, .9)),
                'e_recompute_matches_saved_top1': agreement}
            if agreement < min_e_agreement:
                raise ValueError(f'Evidence-view recomputation disagrees with saved Teacher '
                                 f'targets ({agreement:.3f}); chat framing mismatch')
            outputs[arm] = converted
    finally:
        del cache
        torch.cuda.empty_cache()
    audit['seconds'] = time.monotonic() - started
    return outputs, audit
