from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from experiments.value_lr_physical.gain_transfer_probe import capture_gain_transfer
from experiments.value_lr_physical.readout_transfer_probe import capture_first_answer_readout
from experiments.value_lr_physical.value_alignment_probe import (
    _sampled_write, capture_value_alignment, compare_value_samples, selected_token_pairs,
)
from masked_ttt import MaskedTTT, Request
from masked_ttt._layout import selected_read_plan
from masked_ttt.kv_only import kv_only_cartridge
from masked_ttt.memory import KVReadMemory
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def test_value_samples_compare_same_banks_and_real_dtype_writes():
    native = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.bfloat16)
    basis = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]])
    coefficients = torch.tensor([[[[.2, 0., 0.], [0., .2, 0.]],
                                  [[.1, 0., 0.], [0., .1, 0.]]]])
    banks = torch.tensor([0, 1])
    same = compare_value_samples(native, native.clone(), basis, coefficients, banks, .8)
    assert same["sampled_tokens"] == 2 and same["kv_heads"] == 1
    assert same["median_native_cosine"] == pytest.approx(1.)
    assert same["median_basis_coordinate_cosine"] == pytest.approx(1.)
    assert same["median_written_delta_cosine"] == pytest.approx(1.)
    assert same["full_written_relative_l2"] == pytest.approx(same["selected_written_relative_l2"])
    opposite = compare_value_samples(native, -native, basis, coefficients, banks, .8)
    assert opposite["median_native_cosine"] == pytest.approx(-1.)
    assert opposite["median_written_delta_cosine"] == pytest.approx(-1.)
    rounded = compare_value_samples(native, native.clone(), basis, coefficients * 1e-7, banks, .8)
    assert rounded["full_changed_fraction"] == rounded["selected_changed_fraction"] == 0
    assert rounded["median_written_delta_cosine"] is None
    empty = compare_value_samples(native * 0, native * 0, basis, coefficients, banks, .8)
    assert empty["median_native_cosine"] is None and empty["median_full_basis_energy"] is None
    assert empty["full_written_relative_l2"] is None
    with pytest.raises(ValueError):
        compare_value_samples(native, native.clone(), basis, coefficients, torch.tensor([0, 2]), .8)
    with pytest.raises(ValueError):
        compare_value_samples(native, native.clone(), basis, coefficients, torch.tensor([1, 0]), .8)
    with pytest.raises(ValueError):
        compare_value_samples(native, native.clone(), basis, coefficients, torch.tensor([0., 1.]), .8)


def test_sampled_writer_matches_full_production_writer_on_selected_tokens():
    torch.manual_seed(120)
    config = SimpleNamespace(num_hidden_layers=1, num_key_value_heads=2, num_attention_heads=2,
                             hidden_size=8, head_dim=4, layer_types=["full_attention"])
    chunks = [[2, 3, 4], [5, 6], [7, 8, 9, 10]]
    memory = KVReadMemory(config, chunks, [0, 2], 12, rank=2, sink_tokens=2)
    with torch.no_grad():
        memory.value_coefficients.copy_(torch.randn_like(memory.value_coefficients) * .1)
    native = torch.randn(1, 2, 12, 4).to(torch.bfloat16)
    indices = torch.tensor(memory.source_positions)
    banks = list(memory.source_banks)
    full = memory._chunk_values(0, native)[0].index_select(1, indices)
    sampled = _sampled_write(native[0].index_select(1, indices), memory.value_basis[0],
                             memory.value_coefficients[0], banks, memory.value_radius)
    assert torch.equal(full, sampled)


def test_sampled_writer_preserves_nonconsecutive_chunk_banks():
    native = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]], dtype=torch.float32)
    basis = torch.tensor([[[1.], [0.]]])
    coefficients = torch.tensor([[[[.1, 0.]], [[9., 9.]], [[0., .2]]]])
    actual = _sampled_write(native, basis, coefficients, [0, 2, 2], 1.)
    expected = torch.tensor([[[1.1, 2.], [3., 4.6], [5., 7.]]])
    torch.testing.assert_close(actual, expected)


def test_token_alignment_preserves_original_identity_and_excludes_sinks():
    ids = list(range(30))
    chunks = [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    layout = SimpleNamespace(ids=ids, chunks=chunks, initial=3)
    config = SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1, num_attention_heads=1,
                             hidden_size=4, head_dim=4, layer_types=["full_attention"])
    memory = KVReadMemory(config, chunks, [0, 2], 15, rank=2, sink_tokens=4)
    engine = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [40, 41]),
                             config=SimpleNamespace(sink_tokens=4))
    plan = selected_read_plan(ids, chunks, 3, [0, 2], [40, 41])
    teacher = SimpleNamespace(document_ids=plan.ids[:plan.document_end])
    all_pairs, sampled = selected_token_pairs(engine, layout, [0, 2], teacher, memory, 3)
    assert all_pairs == [(4, 4, 0), (5, 5, 0), (9, 8, 1), (10, 9, 1), (11, 10, 1)]
    assert sampled == [all_pairs[0], all_pairs[2], all_pairs[-1]]
    teacher.document_ids = (999,)
    with pytest.raises(ValueError):
        selected_token_pairs(engine, layout, [0, 2], teacher, memory)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_alignment_observer_preserves_all_outputs_and_releases_caches(architecture):
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
         capture_value_alignment(max_positions=16) as alignment, kv_only_cartridge():
        observed = method.answer(request)
    assert baseline["generated_token_ids"] == observed["generated_token_ids"]
    assert baseline["teacher"]["tokens"] == observed["teacher"]["tokens"]
    assert len(gain) == len(baseline["training"]["steps"]) == 2
    for before, after in zip(baseline["training"]["steps"], observed["training"]["steps"]):
        assert before["student_rollout_tokens"] == after["student_rollout_tokens"]
        assert math.isclose(before["teacher_KL_before"], after["teacher_KL_before"], rel_tol=1e-6, abs_tol=1e-8)
        assert math.isclose(before["value_gradient_norm"], after["value_gradient_norm"], rel_tol=1e-6, abs_tol=1e-8)
    assert set(readout) == set(alignment) == {"statistics"}
    stats = alignment["statistics"]
    assert "unavailable" not in stats and stats["sampled_tokens"] <= 16 and stats["layers"]
    assert stats["reference_answers_used"] is False and stats["model_or_cache_modified"] is False
    assert observed["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    artifact = observed["standalone_cartridge"]["memory"]
    assert artifact["additional_source_kv_bytes_before_decode"] == 0
    assert artifact["additional_training_adapter_bytes_before_decode"] == 0
    assert artifact["kv_only_materialized"] and artifact["cartridge_adapter_bytes"] == 0


def test_alignment_observer_restores_patches_on_error():
    import masked_ttt.pipeline as pipeline

    original = (pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader,
                KVReadMemory.select_actual_read_path)
    with pytest.raises(RuntimeError, match="intentional"):
        with capture_value_alignment():
            raise RuntimeError("intentional")
    assert (pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader,
            KVReadMemory.select_actual_read_path) == original
