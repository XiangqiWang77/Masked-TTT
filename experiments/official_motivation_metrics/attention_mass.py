"""Reconstruct the deployed last-query attention mass on labeled positions.

The computation observes a frozen forward and does not change its outputs.
It includes the actual causal prefix and native local-window restrictions.
"""
from contextlib import contextmanager

import torch
from torch.nn import functional as F


@torch.no_grad()
def target_attention_mass(q, previous_keys, new_keys, start_position, target_positions,
                          *, scale, sliding_window=None, block_size=4096):
    """Mean over heads of the last query's summed probability on target tokens."""
    if q.ndim != 4 or previous_keys.ndim != 4 or new_keys.ndim != 4:
        raise ValueError('Expected Q/K tensors in batch-head-token-dimension layout')
    if q.shape[0] != 1 or previous_keys.shape[0] != 1 or new_keys.shape[0] != 1:
        raise ValueError('This diagnostic operates on one request')
    heads = q.shape[1]
    if heads % previous_keys.shape[1] or new_keys.shape[1] != previous_keys.shape[1]:
        raise ValueError('Query and KV heads are incompatible')
    if new_keys.shape[-2] != q.shape[-2]:
        raise ValueError('Each query position must have a new key')
    query = q[..., -1:, :].float()
    padded = previous_keys.shape[-1]
    if padded < query.shape[-1] or new_keys.shape[-1] != query.shape[-1]:
        raise ValueError('Unexpected native or compiled key width')
    if padded > query.shape[-1]:
        # Compiled score bias: the query coordinate is 1 and new K is 0.
        query = F.pad(torch.cat((query, torch.ones_like(query[..., :1])), dim=-1),
                      (0, padded-query.shape[-1]-1))
        new_keys = F.pad(new_keys, (0, padded-new_keys.shape[-1]))
    groups = heads // previous_keys.shape[1]

    def scores(keys):
        scores_parts = []
        for first in range(0, keys.shape[-2], block_size):
            block = keys[..., first:first+block_size, :].float()
            if groups != 1:
                block = block.repeat_interleave(groups, dim=1)
            scores_parts.append((query @ block.transpose(-1, -2)).squeeze(-2) * scale)
        return torch.cat(scores_parts, dim=-1)

    old_length = previous_keys.shape[-2]
    if old_length < 1:
        raise ValueError('Expected a prefetched document cache')
    logits = torch.cat((scores(previous_keys), scores(new_keys)), dim=-1)
    # The last query sees every earlier position in the current call. Native
    # sliding attention retains exactly the most recent `window` positions.
    absolute_first = int(start_position) - old_length
    if absolute_first < 0:
        raise ValueError('Cache has more keys than preceding positions')
    if sliding_window is not None and logits.shape[-1] > sliding_window:
        dropped = logits.shape[-1] - sliding_window
        logits = logits[..., dropped:]
        absolute_first += dropped
    attention = logits.softmax(dim=-1)
    last = absolute_first + logits.shape[-1]
    offsets = sorted(set(int(i)-absolute_first for i in target_positions if absolute_first <= int(i) < last))
    if not offsets:
        return 0.0
    idx = torch.tensor(offsets, dtype=torch.long, device=attention.device)
    return float(attention.index_select(-1, idx).sum(-1).mean().item())


class AttentionCapture:
    def __init__(self, model, target_positions):
        self.model = model
        self.targets = tuple(sorted(set(int(x) for x in target_positions)))
        self.records = []
        self.handles = []

    def _observe(self, module, args, kwargs):
        from masked_ttt.architectures import project_qkv
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        embeds = kwargs.get('position_embeddings')
        cache = kwargs.get('past_key_values')
        positions = kwargs.get('cache_position')
        if hidden is None or embeds is None or cache is None or positions is None:
            raise ValueError('Missing deployed attention inputs in probe')
        if getattr(module, 'sinks', None) is not None:
            raise ValueError('This OLMo diagnostic does not implement attention sinks')
        old = cache.layers[module.layer_idx].keys
        if old is None:
            raise ValueError('Probe requires a prefilled document')
        q, k, _ = project_qkv(module, hidden, embeds)
        mass = target_attention_mass(q, old, k, int(positions[0]), self.targets,
            scale=module.scaling, sliding_window=module.sliding_window)
        self.records.append(dict(layer=int(module.layer_idx), mass=mass))

    def __enter__(self):
        self.handles = [block.self_attn.register_forward_pre_hook(self._observe, with_kwargs=True)
                        for block in self.model.layers]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def result(self, steps):
        layers = len(self.model.layers)
        if len(self.records) != steps*layers:
            raise ValueError('An attention layer or answer step was not captured')
        output = []
        for step in range(steps):
            batch = self.records[step*layers:(step+1)*layers]
            if [r['layer'] for r in batch] != list(range(layers)):
                raise ValueError('Unexpected attention layer ordering')
            output.append(sum(r['mass'] for r in batch)/layers)
        return dict(attention_mass=sum(output)/len(output), per_answer_step=output,
                    answer_steps=steps, layers=layers, target_positions=len(self.targets))
