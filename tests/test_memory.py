from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from masked_ttt import Backbone, TTTConfig
from masked_ttt.attention import address_attention, bounded_value_residual
from masked_ttt.cache import AppendOnlyDocumentCache, fork_prefix
from masked_ttt.checkpointing import layer_checkpointed_logits
from masked_ttt.memory import KVReadMemory
from masked_ttt.reader import MemoryReader, matched_incremental_logits
from masked_ttt.runtime import private_forward


@pytest.mark.parametrize("group_size", [2, 4])
def test_qwen_grouped_prefill_and_zero_adapter_match_the_frozen_model(tiny_backbone, group_size):
    engine = tiny_backbone
    if group_size == 4:
        # Match Qwen3-4B's four query heads per KV head.
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=1, head_dim=8,
                             max_position_embeddings=256, pad_token_id=0, eos_token_id=2)
        config._attn_implementation = "eager"
        engine = Backbone(Qwen3ForCausalLM(config).eval(), SimpleNamespace(eos_token_id=2),
                          TTTConfig(max_context=256, context_reserve_tokens=32, value_rank=2))
    tokens = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    with torch.no_grad():
        expected = engine.model(input_ids=tokens[None], use_cache=False).logits
        cache = AppendOnlyDocumentCache(engine.model.config, len(tokens))
        values = []
        for start, end in ((0, 3), (3, 7), (7, 10)):
            out = private_forward(engine.model, tokens[start:end], start, cache)
            values.append(engine.model.lm_head(out.last_hidden_state))
        actual = torch.cat(values, 1)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        memory = KVReadMemory(engine.model.config, [list(range(5)), list(range(5, 10))],
                              [1], 10, rank=2, sink_tokens=1)
        memory.prepare_value_basis(cache, memory.adapted_layers)
        memory.value_enabled, memory.native_mode = True, False
        reader = MemoryReader(engine, memory)
        suffix = torch.tensor([15, 16, 17])
        base = private_forward(engine.model, suffix, 10, fork_prefix(cache, 10)).last_hidden_state
        augmented = private_forward(reader.reader, suffix, 10,
                                    reader.compiled_branch(cache)).last_hidden_state
    torch.testing.assert_close(base, augmented, atol=2e-7, rtol=2e-5)


def test_attention_and_all_derivatives_match_direct_GQA_formula():
    torch.manual_seed(12)
    q = torch.randn(1, 4, 3, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 7, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    bias = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    scores = q @ repeat_kv(k, 2).transpose(-1, -2) / (8**0.5)
    scores = scores + bias.repeat_interleave(2, 0)[None, :, None, :]
    visible = torch.arange(7)[None] <= torch.arange(4, 7)[:, None]
    expected = scores.masked_fill(~visible[None, None], float("-inf")).softmax(-1) @ repeat_kv(v, 2)
    actual = address_attention(q, k, v, bias, backend="math")
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(actual.square().sum(), (q, k, v, bias), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (q, k, v, bias))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)


def test_future_document_values_remain_invisible():
    q = torch.zeros(1, 2, 2, 8)
    k = torch.zeros(1, 1, 4, 8)
    values = torch.zeros_like(k)
    values[:, :, 1] = 1
    values[:, :, 3] = 100
    result = address_attention(q, k, values, torch.tensor([[0.0, 5.0, 0.0, 100.0]]), backend="math")
    assert float(result[0, 0, 0, 0]) < 1  # The very strong future position cannot leak.


def test_value_changes_are_masked_and_norm_bounded():
    torch.manual_seed(13)
    values = torch.randn(1, 2, 7, 8)
    basis = torch.randn(2, 8, 2)
    coefficients = torch.randn(2, 2, 8, requires_grad=True) * 100
    selected = torch.tensor([False, True, False, True, False, False, True])
    delta = bounded_value_residual(values, basis, coefficients, selected, 0.15)
    assert not bool(delta[:, :, ~selected].any())
    assert bool((delta.norm(dim=-1) <= 0.15 * values.norm(dim=-1) + 1e-6).all())
    (gradient,) = torch.autograd.grad((delta * torch.randn_like(delta)).sum(), coefficients)
    assert float(gradient.norm()) > 0


def test_zero_state_is_bitwise_native_and_original_cache_is_immutable(memory_fixture):
    f = memory_fixture
    before = [(item.keys.clone(), item.values.clone()) for item in f.cache.layers]
    f.memory.select_actual_read_path()
    with torch.no_grad():
        suffix = torch.tensor([9, 10])
        native = private_forward(
            f.engine.model, suffix, 6, fork_prefix(f.cache, 6)
        ).last_hidden_state
        adapted = private_forward(
            f.reader.reader, suffix, 6, f.reader.compiled_branch(f.cache)
        ).last_hidden_state
    torch.testing.assert_close(native, adapted, atol=0, rtol=0)
    for item, (k, v) in zip(f.cache.layers, before):
        torch.testing.assert_close(item.keys, k, atol=0, rtol=0)
        torch.testing.assert_close(item.values, v, atol=0, rtol=0)
    with torch.no_grad():
        f.memory.gain.fill_(4)
        f.memory.constrain()
    assert not bool(f.memory.biases(0, 9)[:, :3].any())
    assert not bool(f.memory.biases(0, 9)[:, 6:].any())


def test_teacher_forcing_never_reads_the_current_or_future_target(memory_fixture):
    f = memory_fixture
    f.memory.native_mode = False
    with torch.no_grad():
        f.memory.gain[:, :, 1] = 0.1
        a = matched_incremental_logits(
            f.engine, f.reader, f.cache, torch.tensor([9, 10]), torch.tensor([11, 12, 13])
        )
        b = matched_incremental_logits(
            f.engine, f.reader, f.cache, torch.tensor([9, 10]), torch.tensor([11, 12, 77])
        )
        c = matched_incremental_logits(
            f.engine, f.reader, f.cache, torch.tensor([9, 10]), torch.tensor([11, 77, 13])
        )
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a[:2], c[:2], atol=0, rtol=0)
    assert not torch.equal(a[2], c[2])


def test_layer_recomputation_preserves_logits_and_both_gradients(memory_fixture):
    f = memory_fixture
    with torch.no_grad():
        f.memory.gain[:, :, 1] = 0.1
        f.memory.value_coefficients.fill_(0.002)
    f.memory.native_mode = False
    suffix, targets = torch.tensor([9, 10]), torch.tensor([11, 12, 13])
    a = matched_incremental_logits(f.engine, f.reader, f.cache, suffix, targets)
    params = (f.memory.gain, f.memory.value_coefficients)
    ga = torch.autograd.grad(a.square().sum(), params)
    b = layer_checkpointed_logits(f.engine, f.reader, f.cache, suffix, targets)
    gb = torch.autograd.grad(b.square().sum(), params)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    for left, right in zip(ga, gb):
        assert float(left.norm()) > 0
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)
    assert not bool(ga[0][:, :, ~f.memory.selected_chunk_mask].any())
