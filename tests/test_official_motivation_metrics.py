import math
from types import SimpleNamespace

import torch

from experiments.official_motivation_metrics.attention_mass import target_attention_mass


def test_mass_matches_softmax_over_complete_visible_prefix():
    q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    old = torch.tensor([[[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]])
    new = torch.tensor([[[[0.5, 0.0], [-1.0, 0.0]]]])
    logits = torch.tensor([0.0, 1.0, 2.0, 0.5, -1.0])
    expected = float(logits.softmax(0)[[0, 2]].sum())
    actual = target_attention_mass(q, old, new, 3, [0, 2], scale=1.0, block_size=2)
    assert math.isclose(actual, expected, rel_tol=1e-6)


def test_sliding_window_and_missing_target():
    q = torch.ones(1, 1, 1, 1)
    old = torch.zeros(1, 1, 4, 1)
    new = torch.zeros(1, 1, 1, 1)
    assert target_attention_mass(q, old, new, 4, [0], scale=1.0, sliding_window=2) == 0.0
    assert math.isclose(target_attention_mass(q, old, new, 4, [3], scale=1.0, sliding_window=2), 0.5)
    assert target_attention_mass(q, old, new, 4, [], scale=1.0) == 0.0


def test_materialized_score_bias_is_included():
    q = torch.zeros(1, 1, 1, 1)
    old = torch.tensor([[[[0.0, 2.0], [0.0, 0.0]]]])
    new = torch.zeros(1, 1, 1, 1)
    expected = math.exp(2.0) / (math.exp(2.0) + 2.0)
    actual = target_attention_mass(q, old, new, 2, [0], scale=1.0)
    assert math.isclose(actual, expected, rel_tol=1e-6)


def test_gqa_heads_are_averaged():
    q = torch.ones(1, 2, 1, 1)
    old = torch.tensor([[[[1.0], [0.0]]]])
    new = torch.zeros(1, 1, 1, 1)
    expected = math.e / (math.e + 2.0)
    assert math.isclose(target_attention_mass(q, old, new, 2, [0], scale=1.0), expected, rel_tol=1e-6)


def test_tiny_olmo_probe_does_not_change_hidden_states():
    from transformers import Olmo3Config, Olmo3ForCausalLM
    from masked_ttt.architectures import install_hybrid_attention
    from masked_ttt.cache import fork_prefix
    from masked_ttt.config import TTTConfig
    from masked_ttt.model import Backbone
    from masked_ttt.runtime import private_forward, private_module_tree
    from masked_ttt.teacher import prefill_document
    from experiments.official_motivation_metrics.attention_mass import AttentionCapture

    config = Olmo3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention'])
    model = Olmo3ForCausalLM(config).eval()
    engine = Backbone(model, SimpleNamespace(eos_token_id=2), TTTConfig(
        max_context=128, context_reserve_tokens=8, prefill_chunk_tokens=16))
    ids = [1, 3, 4, 5, 6, 7, 8, 9, 10]
    cache = prefill_document(engine, engine.model, ids, 2, [6])

    def reader():
        tree = private_module_tree(model.model)
        install_hybrid_attention(tree)
        return SimpleNamespace(model=tree, lm_head=model.lm_head)

    baseline, observed = reader(), reader()
    suffix = torch.tensor(ids[6:])
    with torch.no_grad():
        expected = private_forward(baseline, suffix, 6, fork_prefix(cache, 6)).last_hidden_state
        with AttentionCapture(observed.model, [2, 4]) as capture:
            actual = private_forward(observed, suffix, 6, fork_prefix(cache, 6)).last_hidden_state
    assert torch.equal(expected, actual)
    result = capture.result(1)
    assert result['layers'] == 2 and 0 <= result['attention_mass'] <= 1
