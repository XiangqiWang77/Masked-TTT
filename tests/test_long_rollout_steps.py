"""Long-rollout checkpoints stay on the original SD/independent-KV path."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments.anchored_short_verifier.paired_controls import original_question_controls
from experiments.long_rollout_steps.probe import capture_step_snapshots
from experiments.long_rollout_steps.runner import long_steps_runner
from experiments.long_rollout_steps.report import summarize, validate_result
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
import experiments.anchored_short_verifier.paired_controls as paired_controls
import masked_ttt as runtime
from masked_ttt import MaskedTTT, Request
from masked_ttt.linked_selection import LinkedFrozenSelector
from masked_ttt.kv_only import kv_only_cartridge
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_materialization_equivalence import gpt_gqa_engine
from test_public_linked_selection import _source
from test_standalone import make_engine


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_two_four_eight_step_checkpoints_and_final_kv_release(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.eos_ids = {-1}
    request = Request(document="An old note was red. The current note is blue. " * 5,
                      question="What color is the current note?", max_new_tokens=12)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    engine.config = replace(engine.config, steps=2, max_answer_steps=2, student_rollout_tokens=32,
        self_question_count=0, guard_enabled=False, gain_head_mode="kv", selected_transfer_weight=0.,
        value_basis_source="document", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    method = MaskedTTT(engine, Selector())
    with kv_only_cartridge(), original_question_controls():
        baseline = method.answer(request)
    engine.config = replace(engine.config, steps=8, max_answer_steps=8)
    with kv_only_cartridge(), original_question_controls(), capture_step_snapshots() as captured:
        candidate = method.answer(request)
        assert not captured["snapshots"]
    probe = candidate["standalone_cartridge"]["step_probe"]
    assert probe["practice_count"] == 0
    assert probe["diagnostic_snapshots_offloaded_to_cpu"]
    assert probe["diagnostic_snapshots_released_before_final_decode"]
    assert set(probe["checkpoints"]) == {"2", "4", "8"}
    assert probe["checkpoints"]["2"]["tokens"] == baseline["generated_token_ids"]
    assert probe["checkpoints"]["8"]["tokens"] == candidate["generated_token_ids"]
    assert len(candidate["training"]["steps"]) == 8
    assert candidate["training"]["student_rollout_limit"] == 32
    assert candidate["training"]["gain_head_mode"] == "kv"
    assert candidate["training"]["selected_transfer_weight"] == 0
    assert candidate["training"]["objective"] == "on-policy KL(teacher || student)"
    assert all(len(step["student_rollout_tokens"]) == 12 for step in candidate["training"]["steps"])
    assert candidate["backbone_unchanged"] and candidate["original_document_KV_unchanged"]
    final = candidate["standalone_cartridge"]["memory"]
    assert final["kv_only_materialized"]
    assert final["additional_source_kv_bytes_before_decode"] == 0
    assert final["additional_training_adapter_bytes_before_decode"] == 0
    assert candidate["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0


def test_frozen_runner_keeps_the_long_rollout_and_releases_hooks(tmp_path):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine()
    engine.eos_ids = {-1}
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda value: value,
        validate=lambda _result, _entry: None, replace=replace, preflight=lambda _engine: None)
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
           paired_controls.original_question_controls)
    with long_steps_runner(implementation, runtime, paired_controls, root="pilot",
                           linked_selector=LinkedFrozenSelector,
                           physical_controls=kv_only_original_question_controls):
        manifest = dict(selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25,
                        student_rollout_tokens=32)
        assert implementation.validate_manifest(manifest)["selection_policy"] == "cached_decision_score10"
        engine.config = implementation.replace(engine.config, max_context=2048, keep_fraction=.1,
            student_rollout_tokens=32, self_question_count=0, guard_enabled=False,
            selected_transfer_weight=0., value_basis_source="document",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        selector = runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with paired_controls.original_question_controls():
            result = MaskedTTT(engine, selector).answer(Request(
                document=document, question=question, max_new_tokens=8))
        implementation.validate(result, {"key": "case"})
        assert result["training"]["gain_head_mode"] == "kv"
        assert len(result["training"]["steps"]) == 8
        result.update(key="case", model="Qwen/Qwen3-4B-Instruct-2507", task="qa1", keep_fraction=.1)
        result["self_judge"]["native_prediction"] = "blue"
        validate_result(result, "case", "qwen", "qa1")
    assert (implementation.ROOT, implementation.validate_manifest, implementation.validate,
            implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
            paired_controls.original_question_controls) == old


def test_step_summary_counts_paired_improvements_and_regressions():
    rows = [dict(case="babilong-8-qa3-1", whole=0, zero=0, teacher=0, step2=0, step4=1, step8=1),
            dict(case="babilong-16-qa3-1", whole=1, zero=1, teacher=1, step2=1, step4=1, step8=0)]
    summary = summarize(rows)
    assert summary["n"] == 2 and summary["source_questions"] == 1
    assert summary["step4_vs_step2"]["new_correct"] == 1
    assert summary["step8_vs_step2"]["new_correct"] == 1
    assert summary["step8_vs_step2"]["new_wrong"] == 1


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
@pytest.mark.parametrize("fixed_step", [2, 4])
def test_fixed_earlier_step_is_the_physically_released_final_cartridge(architecture, fixed_step):
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.eos_ids = {-1}
    request = Request(document="Old note red. Current note blue. " * 5,
                      question="Which color is current?", max_new_tokens=4)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    engine.config = replace(engine.config, steps=fixed_step, max_answer_steps=fixed_step,
        student_rollout_tokens=32, self_question_count=0, guard_enabled=False,
        gain_head_mode="kv", selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    method = MaskedTTT(engine, Selector())
    with kv_only_cartridge(), original_question_controls():
        baseline = method.answer(request)

    engine.config = replace(engine.config, steps=8, max_answer_steps=8)
    with kv_only_cartridge(), original_question_controls(), capture_step_snapshots(
            final_base_step=fixed_step) as captured:
        result = method.answer(request)
        assert not captured["snapshots"]
    final = result["standalone_cartridge"]
    probe = final["step_probe"]
    assert probe["fixed_final_base_step"] == fixed_step
    assert probe["checkpoints"][str(fixed_step)]["tokens"] == result["generated_token_ids"]
    assert result["generated_token_ids"] == baseline["generated_token_ids"]
    assert final["original_question_controls"]["adapted"]["tokens"] == result["generated_token_ids"]
    assert len(result["training"]["steps"]) == 8
    assert result["training"]["objective"] == "on-policy KL(teacher || student)"
    assert result["training"]["selected_transfer_weight"] == 0
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    assert final["memory"]["kv_only_materialized"]
    assert final["memory"]["additional_source_kv_bytes_before_decode"] == 0
    assert final["memory"]["additional_training_adapter_bytes_before_decode"] == 0


@pytest.mark.parametrize("base_steps,fixed_step", [((2, 4, 8), 3), ((4, 2), 2), ((2, 2), 2), ((0, 2), 2)])
def test_invalid_fixed_checkpoint_is_rejected_before_a_request(base_steps, fixed_step):
    with pytest.raises(ValueError):
        with capture_step_snapshots(base_steps, final_base_step=fixed_step):
            pass


@pytest.mark.parametrize("architecture,fixed_step", [("qwen3", 2), ("qwen3", 4),
                                                      ("gpt_oss_gqa", 4), ("gpt_oss_gqa", 8)])
def test_loaded_model_preflight_checks_fixed_checkpoint_and_release(architecture, fixed_step):
    from experiments.long_rollout_steps.runner import preflight
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    old = engine.config
    result = preflight(engine, final_base_step=fixed_step)
    assert result["passed"] and result["fixed_final_base_step"] == fixed_step
    assert result["same_baseline_checkpoint_tokens"] and result["same_fixed_final_tokens"]
    assert result["both_gradient_paths"] and result["independent_cartridge"]
    assert engine.config is old


@pytest.mark.parametrize("practice_count", [1, 2])
def test_checkpoint_budget_includes_verified_practice_and_still_releases_cache(monkeypatch, practice_count):
    import masked_ttt.pipeline as pipeline
    import masked_ttt.self_study_runtime as study
    from masked_ttt.self_study import practice_request

    engine = make_engine()
    engine.config = replace(engine.config, self_question_count=practice_count, guard_enabled=False,
        steps=2, max_answer_steps=2, student_rollout_tokens=8, selected_transfer_weight=0.,
        value_basis_source="document", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request = Request(document="Old note is red. Current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    def practices(engine, teacher_reader, teacher, item, layout, selected):
        paths = []
        for index, question in enumerate(["What is the old color?", "What is the latest color?"][:practice_count]):
            practice = practice_request(item, question, engine.config.student_rollout_tokens)
            student, focused = study.suffixes(engine, layout, selected, teacher, practice)
            paths.append(dict(name="practice_" + str(index + 1), student_suffix=student,
                              teacher=focused, question=question))
        return paths, dict(verified=practice_count, checks=[], reference_answers_used=False,
                           student_answer_hint_used=False)

    monkeypatch.setattr(pipeline, "generate_self_study", practices)
    method = MaskedTTT(engine, Selector())
    with kv_only_cartridge(), original_question_controls():
        baseline = method.answer(request)
    engine.config = replace(engine.config, steps=8, max_answer_steps=8)
    with kv_only_cartridge(), original_question_controls(), capture_step_snapshots(final_base_step=2) as captured:
        candidate = method.answer(request)
        assert not captured["snapshots"]
    probe = candidate["standalone_cartridge"]["step_probe"]
    assert probe["practice_count"] == practice_count
    assert [probe["checkpoints"][str(n)]["total_steps"] for n in (2, 4, 8)] == [
        n + practice_count for n in (2, 4, 8)]
    assert probe["checkpoints"]["2"]["tokens"] == baseline["generated_token_ids"]
    assert probe["fixed_final_base_step"] == 2
    assert probe["checkpoints"]["2"]["tokens"] == candidate["generated_token_ids"]
    memory = candidate["standalone_cartridge"]["memory"]
    assert memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0
