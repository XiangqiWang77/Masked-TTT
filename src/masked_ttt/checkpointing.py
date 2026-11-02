"""Execution-only layer recomputation for the frozen document-memory reader.

Keep the native document KV resident. Materialize one layer's augmented KV
inside its checkpoint, and retain only new suffix KV between answer steps.
The forward arithmetic and causal suffix dependencies are unchanged; no
teacher token, attention layer, or prefix gradient is dropped.
"""

import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .cache import fork_prefix


def layer_checkpointed_logits(engine, reader, cache, suffix, targets, target_positions=None):
    if not torch.is_grad_enabled():
        raise ValueError("This execution overlay is only for gradient passes")
    memory = reader.memory
    model = reader.reader.model
    if memory.native_mode or memory.capture_attention:
        raise ValueError("Require the original augmented derivative path without observation")
    if model.config.model_type != "qwen3" or model.has_sliding_layers:
        raise ValueError("Only the current full-attention Qwen3 operator is supported")
    if suffix.ndim != 1 or targets.ndim != 1 or not len(suffix) or not len(targets):
        raise ValueError("Nonempty one-dimensional suffix and targets required")
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("The backbone must remain frozen")
    length = memory.document_length
    layers = model.layers[: model.config.num_hidden_layers]
    history = [[] for _ in layers]
    values = []
    d = memory.dim
    padded = ((d + 1 + 31) // 32) * 32
    for index in range(len(targets)):
        engine.budget.check()
        token_ids = suffix if index == 0 else targets[index - 1 : index]
        start = length if index == 0 else length + len(suffix) + index - 1
        positions = torch.arange(start, start + len(token_ids), device=token_ids.device)
        hidden = model.embed_tokens(token_ids[None])
        embeddings = model.rotary_emb(hidden, positions[None])
        for layer_index, block in enumerate(layers):
            past = tuple(x for pair in history[layer_index] for x in pair)

            def step(
                hidden,
                *past,
                block=block,
                layer_index=layer_index,
                positions=positions,
                embeddings=embeddings,
                width=len(token_ids),
            ):
                engine.budget.check()
                branch = fork_prefix(cache, length)
                if hasattr(branch, "_document_storage_owner"):
                    del branch._document_storage_owner
                item = branch.layers[layer_index]
                k, v = item.keys, item.values
                bias = memory.biases(layer_index, k.shape[-2])
                component = bias.to(k.dtype)[None, ..., None] * math.sqrt(d)
                item.keys = F.pad(torch.cat((k, component), -1), (0, padded - d - 1))
                item.values = F.pad(memory.values(layer_index, v), (0, padded - d))
                branch._masked_ttt_augmented = True
                if past:
                    item.keys = torch.cat([item.keys] + list(past[0::2]), -2)
                    item.values = torch.cat([item.values] + list(past[1::2]), -2)
                output = block(
                    hidden,
                    attention_mask=None,
                    position_ids=positions[None],
                    past_key_values=branch,
                    use_cache=True,
                    cache_position=positions,
                    position_embeddings=embeddings,
                )
                return (
                    output,
                    item.keys[..., -width:, :].clone(),
                    item.values[..., -width:, :].clone(),
                )

            result = checkpoint(step, hidden, *past, use_reentrant=False, preserve_rng_state=False)
            hidden = result[0]
            history[layer_index].append(result[1:])
        hidden = model.norm(hidden)
        values.append(engine.model.lm_head(hidden[0, -1:]).float())
    logits = torch.cat(values)
    return logits if target_positions is None else logits.index_select(0, target_positions)
