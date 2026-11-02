"""One causal reader for gradient updates, evaluation, and final decoding."""

import math
from types import MethodType, SimpleNamespace
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from .attention import native_attention, address_attention
from .cache import fork_prefix
from .runtime import private_module_tree, private_forward


class MemoryReader:
    def __init__(self, engine, memory):
        self.engine = engine
        self.memory = memory
        self.reader = SimpleNamespace(
            model=private_module_tree(engine.model.model), lm_head=engine.model.lm_head
        )
        for block in self.reader.model.layers:
            attn = block.self_attn
            native = attn.forward

            def forward(
                module,
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                local_native=native,
                **kwargs,
            ):
                if memory.native_mode and (not memory.capture_attention):
                    return local_native(
                        hidden_states=hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        **kwargs,
                    )
                shape = (*hidden_states.shape[:-1], -1, module.head_dim)
                q = module.q_norm(module.q_proj(hidden_states).view(shape)).transpose(1, 2)
                k = module.k_norm(module.k_proj(hidden_states).view(shape)).transpose(1, 2)
                v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
                (cos, sin) = position_embeddings
                (q, k) = apply_rotary_pos_emb(q, k, cos, sin)
                augmented = getattr(past_key_values, "_masked_ttt_augmented", False)
                if augmented:
                    d = module.head_dim
                    padded = (d + 1 + 31) // 32 * 32
                    (k, v) = (F.pad(k, (0, padded - d)), F.pad(v, (0, padded - d)))
                if past_key_values is not None:
                    (k, v) = past_key_values.update(
                        k,
                        v,
                        module.layer_idx,
                        dict(sin=sin, cos=cos, cache_position=cache_position),
                    )
                if memory.capture_attention:
                    memory.observe(module.layer_idx, q, k)
                if memory.native_mode:
                    result = native_attention(q, k, v, scale=module.scaling)
                elif augmented:
                    qp = F.pad(
                        torch.cat((q, torch.ones_like(q[..., :1])), -1),
                        (0, k.shape[-1] - module.head_dim - 1),
                    )
                    result = native_attention(qp, k, v, scale=module.scaling)[
                        ..., : module.head_dim
                    ]
                else:
                    result = address_attention(
                        q,
                        k,
                        memory.values(module.layer_idx, v),
                        memory.biases(module.layer_idx, k.shape[-2]),
                    )
                result = result.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
                return (module.o_proj(result), None)

            attn.forward = MethodType(forward, attn)

    def compiled_branch(self, cache):
        branch = fork_prefix(cache, self.memory.document_length)
        if self.memory.native_mode:
            return branch
        if hasattr(branch, "_document_storage_owner"):
            del branch._document_storage_owner
        d = self.memory.dim
        for layer, item in enumerate(branch.layers):
            (k, v) = (item.keys, item.values)
            bias = self.memory.biases(layer, k.shape[-2])
            component = bias.to(k.dtype)[None, ..., None] * math.sqrt(d)
            padded = (d + 1 + 31) // 32 * 32
            item.keys = F.pad(torch.cat((k, component), -1), (0, padded - d - 1))
            item.values = F.pad(self.memory.values(layer, v), (0, padded - d))
        branch._masked_ttt_augmented = True
        return branch


def matched_incremental_logits(engine, reader, cache, suffix, targets, target_positions=None):
    """Identical operation order with or without gradient tracking."""
    if suffix.ndim != 1 or targets.ndim != 1 or not len(suffix) or not len(targets):
        raise ValueError("Nonempty one-dimensional suffix and target IDs required")
    if torch.is_grad_enabled():
        return checkpointed_incremental_logits(
            engine, reader, cache, suffix, targets, target_positions
        )
    length = reader.memory.document_length
    branch = reader.compiled_branch(cache)
    engine.budget.check()
    output = private_forward(reader.reader, suffix, length, branch)
    values = []
    for index in range(len(targets)):
        values.append(engine.model.lm_head(output.last_hidden_state[0, -1:]).float())
        if index + 1 < len(targets):
            engine.budget.check()
            output = private_forward(
                reader.reader, targets[index : index + 1], length + len(suffix) + index, branch
            )
    logits = torch.cat(values)
    return logits if target_positions is None else logits.index_select(0, target_positions)


def checkpointed_incremental_logits(engine, reader, cache, suffix, targets, target_positions=None):
    """Recompute attention for backward without saving a document copy per token.

    Each functional step owns its cache. Only the newly added suffix K/V
    blocks survive; old suffix blocks are shared tensor references. Replaying
    a checkpoint cannot append to another step's cache or see future tokens.
    """
    length = reader.memory.document_length
    layers = len(cache.layers)
    blocks = []
    values = []
    for index in range(len(targets)):
        token_ids = suffix if index == 0 else targets[index - 1 : index]
        position = length if index == 0 else length + len(suffix) + index - 1
        history = tuple(x for block in blocks for x in block)

        def step(*past, step_tokens=token_ids, step_position=position):
            engine.budget.check()
            branch = reader.compiled_branch(cache)
            for layer, item in enumerate(branch.layers):
                keys = [item.keys] + list(past[2 * layer :: 2 * layers])
                vals = [item.values] + list(past[2 * layer + 1 :: 2 * layers])
                if len(keys) > 1:
                    item.keys = torch.cat(keys, -2)
                    item.values = torch.cat(vals, -2)
            output = private_forward(reader.reader, step_tokens, step_position, branch)
            additions = tuple(
                x[..., -len(step_tokens) :, :].clone()
                for item in branch.layers
                for x in (item.keys, item.values)
            )
            return (engine.model.lm_head(output.last_hidden_state[0, -1:]).float(), *additions)

        result = checkpoint(step, *history, use_reentrant=False, preserve_rng_state=False)
        values.append(result[0])
        blocks.append(result[1:])
    logits = torch.cat(values)
    return logits if target_positions is None else logits.index_select(0, target_positions)
