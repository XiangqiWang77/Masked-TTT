from dataclasses import replace
import math

import pytest
import torch

from experiments.value_lr_physical.gain_transfer_probe import capture_gain_transfer, chunk_components
from experiments.value_lr_physical.readout_transfer_probe import capture_first_answer_readout
from masked_ttt import MaskedTTT, Request
from masked_ttt.kv_only import kv_only_cartridge
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def test_chunk_projection_is_orthogonal_and_handles_zero_and_single_chunk():
    signal = torch.tensor([[[2., 4., 6.], [-3., -1., 1.]]])
    stats = chunk_components(signal)
    assert stats["selected_chunks"] == 3
    assert stats["total_norm"] == pytest.approx(float(signal.norm()))
    assert stats["common_norm"] ** 2 + stats["contrast_norm"] ** 2 == pytest.approx(stats["total_norm"] ** 2)
    assert 0 < stats["common_squared_fraction"] < 1
    assert chunk_components(torch.zeros(2, 3, 4))["common_squared_fraction"] is None
    one = chunk_components(torch.tensor([[[2.]]]))
    assert one["contrast_norm"] == 0 and one["common_squared_fraction"] == 1
    with pytest.raises(ValueError):
        chunk_components(torch.zeros(2, 3, 0))
    with pytest.raises(ValueError):
        chunk_components(torch.tensor([[[float("nan")]]]))


def test_common_document_gain_cancels_conditionally_but_not_against_suffix():
    document = torch.tensor([-2., 0.5, 1.2], dtype=torch.float64)
    shifted = document + 3.
    torch.testing.assert_close(document.softmax(-1), shifted.softmax(-1))
    with_suffix = torch.cat((document, torch.tensor([0.], dtype=torch.float64))).softmax(-1)
    shifted_with_suffix = torch.cat((shifted, torch.tensor([0.], dtype=torch.float64))).softmax(-1)
    assert shifted_with_suffix[:3].sum() > with_suffix[:3].sum()


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_capture_preserves_training_and_releases_all_tensors(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    engine.config = replace(
        engine.config, steps=2, max_answer_steps=2, self_question_count=0,
        guard_enabled=False, selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True,
    )
    request = Request(
        document="The old note is red. The current note is blue. " * 6,
        question="Which color is current?", max_new_tokens=4,
    )

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    method = MaskedTTT(engine, Selector())
    with kv_only_cartridge():
        baseline = method.answer(request)
    with capture_gain_transfer() as observations, capture_first_answer_readout() as readout, kv_only_cartridge():
        observed = method.answer(request)
    assert baseline["generated_token_ids"] == observed["generated_token_ids"]
    assert baseline["teacher"]["tokens"] == observed["teacher"]["tokens"]
    assert len(observations) == len(baseline["training"]["steps"]) == 2
    for old, new, item in zip(baseline["training"]["steps"], observed["training"]["steps"], observations):
        assert old["student_rollout_tokens"] == new["student_rollout_tokens"]
        for key in ("teacher_KL_before", "gain_gradient_norm", "value_gradient_norm"):
            assert math.isclose(old[key], new[key], rel_tol=1e-6, abs_tol=1e-8)
        assert item["gradient"]["total_norm"] == pytest.approx(old["gain_gradient_norm"], rel=1e-6, abs=1e-8)
        assert item["optimizer_update"]["selected_chunks"] == 2
    assert observed["backbone_unchanged"] and observed["original_document_KV_unchanged"]
    assert observed["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    memory = observed["standalone_cartridge"]["memory"]
    assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0
    assert memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0
    assert set(readout) == {"statistics"}
    statistics = readout["statistics"]
    assert statistics["reference_answers_used"] is False
    assert all(math.isfinite(statistics[key]) and statistics[key] >= 0 for key in (
        "kl_full_native_to_selected_native", "kl_full_native_to_selected_adapted",
        "kl_full_native_to_full_adapted", "kl_selected_native_to_selected_adapted"))


def test_capture_restores_pipeline_after_exception():
    import masked_ttt.pipeline as pipeline

    original = pipeline.KVReadMemory
    with pytest.raises(RuntimeError, match="intentional"):
        with capture_gain_transfer():
            raise RuntimeError("intentional")
    assert pipeline.KVReadMemory is original


def test_readout_capture_restores_pipeline_after_exception():
    import masked_ttt.pipeline as pipeline
    from masked_ttt.memory import KVReadMemory

    original = (pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader,
                KVReadMemory.select_actual_read_path)
    with pytest.raises(RuntimeError, match="intentional"):
        with capture_first_answer_readout():
            raise RuntimeError("intentional")
    assert (pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader,
            KVReadMemory.select_actual_read_path) == original
