import math

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from experiments.gqa_head_gain.attention import augmented_head_bias, head_bias_attention
from masked_ttt.attention import address_attention, native_attention, padded_qkv


@pytest.mark.parametrize("groups", [1, 2, 4, 8])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_each_query_head_bias_matches_direct_gqa_scores_and_gradients(groups, use_sinks):
    torch.manual_seed(53)
    kv_heads, query_positions, key_positions, dim = 2, 3, 7, 8
    query_heads = kv_heads * groups
    q = torch.randn(1, query_heads, query_positions, dim, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, kv_heads, key_positions, dim, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, kv_heads, key_positions, dim, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(query_heads, key_positions, dtype=torch.float64, requires_grad=True)
    sinks = torch.randn(query_heads, dtype=torch.float64) if use_sinks else None
    actual = head_bias_attention(q, k, v, bias, backend="eager", sinks=sinks)

    other_q = q.detach().clone().requires_grad_()
    other_k = k.detach().clone().requires_grad_()
    other_v = v.detach().clone().requires_grad_()
    other_bias = bias.detach().clone().requires_grad_()
    scores = other_q @ repeat_kv(other_k, groups).transpose(-1, -2) / math.sqrt(dim)
    scores = scores + other_bias[None, :, None, :]
    visible = torch.arange(key_positions)[None] <= torch.arange(key_positions - query_positions, key_positions)[:, None]
    scores = scores.masked_fill(~visible[None, None], float("-inf"))
    if sinks is not None:
        scores = torch.cat((scores, sinks[None, :, None, None].expand(1, -1, query_positions, -1)), -1)
    probabilities = scores.softmax(-1)
    if sinks is not None:
        probabilities = probabilities[..., :-1]
    expected = probabilities @ repeat_kv(other_v, groups)
    weight = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weight).sum(), (q, k, bias, v))
    expected_grads = torch.autograd.grad((expected * weight).sum(), (other_q, other_k, other_bias, other_v))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for observed, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed, reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("groups", [1, 4, 8])
def test_shared_bias_reproduces_existing_method_and_zero_is_native(groups):
    torch.manual_seed(59)
    q = torch.randn(1, 2 * groups, 2, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 6, 8, dtype=torch.float64)
    v = torch.randn_like(k)
    shared = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    reference = address_attention(q, k, v, shared, backend="eager")
    expanded = head_bias_attention(q, k, v, shared.repeat_interleave(groups, dim=0), backend="eager")
    torch.testing.assert_close(expanded, reference, atol=1e-12, rtol=1e-12)
    shared_grad = torch.autograd.grad(reference.square().sum(), shared, retain_graph=True)[0]
    expanded_grad = torch.autograd.grad(expanded.square().sum(), shared)[0]
    torch.testing.assert_close(expanded_grad, shared_grad, atol=1e-12, rtol=1e-12)
    zero = head_bias_attention(q, k, v, torch.zeros(2 * groups, 6, dtype=torch.float64), backend="eager")
    torch.testing.assert_close(zero, native_attention(q, k, v, scale=1 / math.sqrt(8), backend="eager"),
                               atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("query_heads,kv_heads,dim,padded", [(32, 8, 128, 160), (64, 8, 64, 96)])
def test_real_backbone_padding_does_not_replicate_kv(query_heads, kv_heads, dim, padded):
    q = torch.zeros(1, query_heads, 1, dim)
    k = torch.zeros(1, kv_heads, 3, dim)
    qp, kp, vp, native = augmented_head_bias(q, k, k, torch.zeros(query_heads, 3))
    _, old_k, old_v, _ = padded_qkv(q, k, k, torch.zeros(kv_heads, 3))
    assert native == dim and qp.shape[-1] == kp.shape[-1] == vp.shape[-1] == padded
    assert kp.shape[1] == vp.shape[1] == kv_heads
    assert padded == ((dim + 1 + 31) // 32) * 32
    assert kp.numel() * kp.element_size() == old_k.numel() * old_k.element_size()
    assert vp.numel() * vp.element_size() == old_v.numel() * old_v.element_size()


def test_invalid_grouping_bias_and_kernel_width_are_rejected():
    k = torch.zeros(1, 2, 3, 8)
    with pytest.raises(ValueError):
        augmented_head_bias(torch.zeros(1, 3, 1, 8), k, k, torch.zeros(3, 3))
    with pytest.raises(ValueError):
        augmented_head_bias(torch.zeros(1, 4, 1, 8), k, k, torch.zeros(2, 3))
    wide = torch.zeros(1, 2, 3, 256)
    with pytest.raises(ValueError):
        augmented_head_bias(torch.zeros(1, 4, 1, 256), wide, wide, torch.zeros(4, 3))
