import pytest
import torch
from experiments.value_lr_physical.zero_path_probe import measure_zero_path_parity
from test_selected_transfer import gpt_engine, setup_views
from test_standalone import make_engine


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_zero_path_probe_preserves_trained_state_and_selected_source(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    _, layout, selected, _, teacher, trained, _ = setup_views(engine)
    with torch.no_grad():
        trained.gain[..., selected[0]] = .2
        trained.value_coefficients.normal_(0, .01)
    trained.value_enabled = True
    trained.native_mode = False
    snapshot = trained.snapshot()
    state = [(layer.keys.clone(), layer.values.clone()) for layer in teacher.document_cache.layers]
    result = measure_zero_path_parity(engine, layout, selected, teacher, trained, max_new_tokens=3)
    assert not result["teacher_answer_used"] and not result["reference_answers_used"]
    assert not result["source_cache_modified"]
    assert result["first_token_kl"] < 1e-5
    assert not result["first_token_top1_changed"]
    assert not result["generated_tokens_changed"]
    assert torch.equal(trained.gain, snapshot["gain"])
    assert torch.equal(trained.value_coefficients, snapshot["value_coefficients"])
    assert trained.native_mode == snapshot["native_mode"]
    for layer, (keys, values) in zip(teacher.document_cache.layers, state):
        torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
        torch.testing.assert_close(layer.values, values, atol=0, rtol=0)
