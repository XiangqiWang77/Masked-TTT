"""GQA attention with an exact additive document-chunk bias."""

import math
import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


def padded_qkv(q, k, v, bias):
    """Add exactly bias[KV-head,key-position] after native RoPE.

    Original scale is retained. Only the first extra Q component is one;
    the matching K component is bias*sqrt(original_head_dim).
    """
    d = q.shape[-1]
    if k.shape[-1] != d or v.shape[-1] != d or bias.shape != k.shape[1:3]:
        raise ValueError("Native Q/K/V dimensions and KV-head address biases required")
    padded = ((d + 1 + 31) // 32) * 32
    if padded > 256:
        raise ValueError("This FlashAttention operator needs native head_dim below256")
    one = torch.ones_like(q[..., :1])
    qp = F.pad(torch.cat((q, one), -1), (0, padded - d - 1))
    component = bias.to(k.dtype)[None, ..., None].expand(k.shape[0], -1, -1, -1)
    kp = F.pad(torch.cat((k, component * math.sqrt(d)), -1), (0, padded - d - 1))
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
