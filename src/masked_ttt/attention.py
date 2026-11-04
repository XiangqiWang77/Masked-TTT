"""GQA attention with an exact additive document-chunk bias."""

import math
import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


def address_dimensions(dim, query_heads, kv_heads, gain_heads):
    """Return the extra address width and the kernel-padded head dimension."""
    if kv_heads < 1 or query_heads % kv_heads or gain_heads not in (kv_heads, query_heads):
        raise ValueError("Address heads must be shared KV heads or individual query heads")
    width = 1 if gain_heads == kv_heads else query_heads // kv_heads
    padded = ((dim + width + 31) // 32) * 32
    if padded > 256:
        raise ValueError("This FlashAttention operator needs native head_dim below256")
    return width, padded


def address_query(q, kv_heads, gain_heads):
    """Append one shared or group-one-hot address to each query head."""
    dim, query_heads = q.shape[-1], q.shape[1]
    width, padded = address_dimensions(dim, query_heads, kv_heads, gain_heads)
    if width == 1:
        address = torch.ones_like(q[..., :1])
    else:
        address = F.one_hot(torch.arange(query_heads, device=q.device) % width, width).to(q.dtype)
        address = address[None, :, None, :].expand(q.shape[0], -1, q.shape[-2], -1)
    return F.pad(torch.cat((q, address), -1), (0, padded - dim - width))


def address_key(k, bias, query_heads):
    """Bake query-specific biases into spare dimensions of shared GQA keys."""
    dim, kv_heads, positions = k.shape[-1], k.shape[1], k.shape[-2]
    if bias.ndim != 2 or bias.shape[-1] != positions:
        raise ValueError("Chunk bias must have one value per head and key position")
    width, padded = address_dimensions(dim, query_heads, kv_heads, bias.shape[0])
    if width == 1:
        component = bias[..., None]
    else:
        component = bias.reshape(kv_heads, width, positions).permute(0, 2, 1)
    component = component.to(device=k.device, dtype=k.dtype)[None].expand(k.shape[0], -1, -1, -1)
    return F.pad(torch.cat((k, component * math.sqrt(dim)), -1), (0, padded - dim - width))


def padded_qkv(q, k, v, bias):
    """Add per-KV-head or per-query-head chunk bias after native RoPE."""
    d = q.shape[-1]
    if k.shape[-1] != d or v.shape != k.shape or q.shape[0] != k.shape[0]:
        raise ValueError("Native Q/K/V dimensions and address biases required")
    _, padded = address_dimensions(d, q.shape[1], k.shape[1], bias.shape[0])
    qp = address_query(q, k.shape[1], bias.shape[0])
    kp = address_key(k, bias, q.shape[1])
    return qp, kp, F.pad(v, (0, padded - d)), d


def native_attention(
    q, k, v, *, scale, backend="auto", causal=True, sliding_window=None, sinks=None
):
    """Bottom-right causal attention, including native GQA."""
    use_flash = backend == "flash" or (backend == "auto" and q.is_cuda)
    if use_flash and sinks is not None and sliding_window is None:
        # A zero-value key implements the learned softmax sink exactly. Each
        # query head carries its own sink logit, so native GQA is preserved.
        width = q.shape[-1]
        padded = ((width + 1 + 31) // 32) * 32
        sink_q = sinks.to(q)[None, :, None, None].expand(q.shape[0], -1, q.shape[-2], -1)
        qp = F.pad(torch.cat((q, sink_q), -1), (0, padded - width - 1))
        kp = F.pad(k, (0, padded - width))
        vp = F.pad(v, (0, padded - width))
        virtual = torch.zeros_like(kp[..., :1, :])
        virtual[..., width] = 1 / scale
        kp = torch.cat((virtual, kp), -2)
        vp = torch.cat((torch.zeros_like(vp[..., :1, :]), vp), -2)
        return native_attention(qp, kp, vp, scale=scale, backend="flash", causal=causal)[
            ..., :width
        ]
    if use_flash and sinks is None:
        from flash_attn import flash_attn_func

        return flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
            deterministic=True,
            window_size=(-1, -1) if sliding_window is None else (sliding_window - 1, 0),
        ).transpose(1, 2)
    groups = q.shape[1] // k.shape[1]
    if groups < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("Q heads must be a multiple of KV heads")
    kk, vv = repeat_kv(k, groups), repeat_kv(v, groups)
    scores = q @ kk.transpose(-1, -2) * scale
    qpos = torch.arange(q.shape[-2], device=q.device) + k.shape[-2] - q.shape[-2]
    visible = torch.arange(k.shape[-2], device=q.device)[None] <= qpos[:, None]
    if sliding_window is not None:
        visible &= torch.arange(k.shape[-2], device=q.device)[None] > qpos[:, None] - sliding_window
    if causal:
        scores = scores.masked_fill(~visible[None, None], float("-inf"))
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    if sinks is not None:
        sink_scores = sinks.to(scores)[None, :, None, None].expand(*scores.shape[:-1], 1)
        scores = torch.cat((scores, sink_scores), -1)
    probabilities = torch.nan_to_num(scores.softmax(-1, dtype=dtype)).to(q.dtype)
    if sinks is not None:
        probabilities = probabilities[..., :-1]
    return probabilities @ vv


def address_attention(
    q, k, v, bias, *, backend="auto", causal=True, sliding_window=None, sinks=None
):
    qp, kp, vp, d = padded_qkv(q, k, v, bias)
    return native_attention(
        qp,
        kp,
        vp,
        scale=1 / math.sqrt(d),
        backend=backend,
        causal=causal,
        sliding_window=sliding_window,
        sinks=sinks,
    )[..., :d]


def bounded_value_residual(values, a, b, selected, rho):
    """Shared rank-r map of native V, with an exact per-position trust region."""
    if not 0 < rho <= 1:
        raise ValueError("Require a positive native-value relative trust radius")
    delta = (values.float() @ a.float()) @ b.float()
    radius = rho * values.float().norm(dim=-1, keepdim=True)
    factor = (radius / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
    delta = delta * factor * selected[None, None, :, None]
    return delta.to(values.dtype)
