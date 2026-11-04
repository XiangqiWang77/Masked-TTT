import pytest
import torch
from dataclasses import replace
from experiments.value_lr_physical.zero_path_probe import capture_zero_path_parity, measure_zero_path_parity
from experiments.value_lr_physical.gain_transfer_probe import capture_gain_transfer
from experiments.value_lr_physical.readout_transfer_probe import capture_first_answer_readout
from experiments.value_lr_physical.value_alignment_probe import capture_value_alignment
from masked_ttt import MaskedTTT, Request
from masked_ttt.kv_only import kv_only_cartridge
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
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


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_observer_composes_with_existing_probes_and_releases_tensors(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    engine.config = replace(engine.config, steps=2, max_answer_steps=2, self_question_count=0,
        guard_enabled=False, selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=4)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    method = MaskedTTT(engine, Selector())
    with kv_only_cartridge():
        baseline = method.answer(request)
    with capture_gain_transfer() as gain, capture_first_answer_readout() as readout, \
         capture_value_alignment(max_positions=16) as alignment, capture_zero_path_parity() as parity, \
         kv_only_cartridge():
        observed = method.answer(request)
    assert baseline["generated_token_ids"] == observed["generated_token_ids"]
    assert baseline["teacher"]["tokens"] == observed["teacher"]["tokens"]
    assert len(gain) == len(baseline["training"]["steps"]) == 2
    assert set(readout) == set(alignment) == set(parity) == {"statistics"}
    assert "unavailable" not in parity["statistics"]
    assert not parity["statistics"]["generated_tokens_changed"]
    assert observed["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    artifact = observed["standalone_cartridge"]["memory"]
    assert artifact["additional_source_kv_bytes_before_decode"] == 0
    assert artifact["additional_training_adapter_bytes_before_decode"] == 0
    assert artifact["kv_only_materialized"] and artifact["cartridge_adapter_bytes"] == 0
