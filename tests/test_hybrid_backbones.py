from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import Olmo3Config, Olmo3ForCausalLM, GptOssConfig, GptOssForCausalLM

from masked_ttt import Backbone, TTTConfig
from masked_ttt.cache import AppendOnlyDocumentCache, fork_prefix
from masked_ttt.memory import KVReadMemory
from masked_ttt.reader import MemoryReader, matched_incremental_logits
from masked_ttt.checkpointing import layer_checkpointed_logits
from masked_ttt.runtime import private_forward


@pytest.fixture(params=["olmo3", "gpt_oss", "gpt_oss_gqa"])
def hybrid(request):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    common = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        pad_token_id=0,
        eos_token_id=2,
    )
    if request.param == "olmo3":
        config = Olmo3Config(**common)
        cls = Olmo3ForCausalLM
    else:
        if request.param == "gpt_oss_gqa":
            # Match GPT-OSS-20B's eight query heads per shared KV head.
            common["hidden_size"] = 64
            common["intermediate_size"] = 128
            common["num_attention_heads"] = 8
            common["num_key_value_heads"] = 1
        config = GptOssConfig(
            **common, head_dim=8, num_local_experts=2, num_experts_per_tok=1, rope_scaling=None
        )
        cls = GptOssForCausalLM
    config._attn_implementation = "eager"
    reference = cls(config).eval()
    if request.param.startswith("gpt_oss"):
        for block in reference.model.layers:
            with torch.no_grad():
                block.self_attn.sinks.fill_(0.2)
    model = deepcopy(reference)
    engine = Backbone(
        model,
        SimpleNamespace(eos_token_id=2),
        TTTConfig(
            max_context=256,
            context_reserve_tokens=32,
            value_rank=2,
            steps=2,
            prefill_chunk_tokens=4,
        ),
    )
    return engine, reference


def test_hybrid_native_causality_and_rolling_cache(hybrid):
    engine, reference = hybrid
    tokens = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    with torch.no_grad():
        expected = reference(input_ids=tokens[None], use_cache=False).logits
        cache = AppendOnlyDocumentCache(engine.model.config, 10)
        logits = []
        for start, end in [(0, 3), (3, 7), (7, 10)]:
            out = private_forward(engine.model, tokens[start:end], start, cache)
            logits.append(engine.model.lm_head(out.last_hidden_state))
    actual = torch.cat(logits, 1)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    assert cache.layers[0].keys.shape[-2] == 3
    assert cache.layers[0].get_seq_length() == 10
    assert cache.layers[1].keys.shape[-2] == 10
    before = [(x._version, x.clone()) for item in cache.layers for x in (item.keys, item.values)]
    with torch.no_grad():
        private_forward(engine.model, torch.tensor([15, 16]), 10, fork_prefix(cache, 10))
    for (version, saved), x in zip(
        before, [x for item in cache.layers for x in (item.keys, item.values)]
    ):
        assert version == x._version
        torch.testing.assert_close(saved, x, atol=0, rtol=0)


def test_hybrid_kv_gradients_checkpoint_and_zero_equivalence(hybrid):
    engine, _ = hybrid
    document = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12])
    cache = AppendOnlyDocumentCache(engine.model.config, len(document))
    with torch.no_grad():
        private_forward(engine.model, document, 0, cache)
    memory = KVReadMemory(
        engine.model.config, [[0, 1, 2, 3], [4, 5, 6, 7]], [1], 8, rank=2, sink_tokens=1
    )
    memory.prepare_value_basis(cache, memory.adapted_layers)
    memory.value_enabled = True
    reader = MemoryReader(engine, memory)
    suffix = torch.tensor([18, 19, 20, 21, 22])
    targets = torch.tensor([23, 24, 25])
    memory.native_mode = True
    with torch.no_grad():
        a = private_forward(engine.model, suffix, 8, fork_prefix(cache, 8)).last_hidden_state
        b = private_forward(
            reader.reader, suffix, 8, reader.compiled_branch(cache)
        ).last_hidden_state
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    memory.native_mode = False
    # The optimizer's first forward uses the differentiable augmented path
    # at zero state; it must still reproduce the unmodified GQA readout.
    with torch.no_grad():
        zero = private_forward(
            reader.reader, suffix, 8, reader.compiled_branch(cache)
        ).last_hidden_state
    torch.testing.assert_close(a, zero, atol=2e-7, rtol=2e-5)
    with torch.no_grad():
        memory.gain[1, :, 1].fill_(0.05)
        memory.value_coefficients[1].fill_(0.001)
    first = matched_incremental_logits(engine, reader, cache, suffix, targets)
    g1 = torch.autograd.grad(first.square().sum(), tuple(memory.parameters()))
    second = layer_checkpointed_logits(engine, reader, cache, suffix, targets)
    g2 = torch.autograd.grad(second.square().sum(), tuple(memory.parameters()))
    torch.testing.assert_close(first, second, atol=1e-7, rtol=1e-5)
    for a, b in zip(g1, g2):
        assert a.isfinite().all() and a[1].abs().max() > 0
        assert a[0].count_nonzero() == 0
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    assert all(p.grad is None and not p.requires_grad for p in engine.model.parameters())
