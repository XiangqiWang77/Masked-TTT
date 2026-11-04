"""Per-query-head chunk biases without replicating shared GQA keys or values.

This is an experimental attention primitive. It does not change the original
OLMo-style training or the production reader until explicitly integrated.
"""
import math

import torch
from torch.nn import functional as F

from masked_ttt.attention import native_attention


def augmented_head_bias(q, k, v, bias):
    """Encode one bias per query head into a shared key's spare dimensions.

    ``bias`` has shape ``[query_heads, key_positions]``. The query head within
    each GQA group receives its own one-hot address; the shared key carries
    that group's bias values. The original softmax scale must be retained.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] != k.shape[0]:
        raise ValueError("Batched attention tensors with shared K/V shape are required")
    batch, query_heads, query_positions, dim = q.shape
    kv_heads, key_positions = k.shape[1:3]
    if k.shape[-1] != dim or kv_heads < 1 or query_heads % kv_heads or bias.shape != (query_heads, key_positions):
        raise ValueError("The head grouping or bias dimensions are invalid")
    groups = query_heads // kv_heads
    padded = ((dim + groups + 31) // 32) * 32
    if padded > 256:
        raise ValueError("The padded head dimension exceeds the attention kernel limit")
    address = F.one_hot(torch.arange(query_heads, device=q.device) % groups, groups).to(q.dtype)
    address = address[None, :, None, :].expand(batch, -1, query_positions, -1)
    qp = F.pad(torch.cat((q, address), dim=-1), (0, padded - dim - groups))
    component = bias.reshape(kv_heads, groups, key_positions).permute(0, 2, 1)
    component = component.to(device=k.device, dtype=k.dtype)[None].expand(batch, -1, -1, -1)
    kp = F.pad(torch.cat((k, component * math.sqrt(dim)), dim=-1), (0, padded - dim - groups))
    return qp, kp, F.pad(v, (0, padded - dim)), dim


def head_bias_attention(q, k, v, bias, *, backend="auto", causal=True, sliding_window=None, sinks=None):
    qp, kp, vp, dim = augmented_head_bias(q, k, v, bias)
    return native_attention(qp, kp, vp, scale=1 / math.sqrt(dim), backend=backend,
                            causal=causal, sliding_window=sliding_window, sinks=sinks)[..., :dim]
