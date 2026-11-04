from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

import masked_ttt as real_runtime
import experiments.anchored_short_verifier.paired_controls as real_controls
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.self_study_prompt.study_runner import study_pair_runner
from masked_ttt import Request
from masked_ttt.config import TTTConfig
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_public_linked_selection import _source
from test_standalone import make_engine


def make_runner():
    old_method, old_selector, old_controls, physical_controls = object(), object(), object(), object()
    runtime = SimpleNamespace(MaskedTTT=old_method, FrozenJSONSelector=old_selector)
    controls = SimpleNamespace(original_question_controls=old_controls)
    events = []

    def manifest(value):
        events.append(("manifest", value["selection_policy"]))
        return 200, value["models"]

    def preflight(_engine):
        assert runtime.MaskedTTT is old_method
        assert controls.original_question_controls is physical_controls
        events.append(("preflight", True))
        return {"passed": True}

    implementation = SimpleNamespace(ROOT="old", validate_manifest=manifest,
        validate=lambda *args: None, replace=replace, preflight=preflight)
    return implementation, runtime, controls, old_method, old_selector, old_controls, physical_controls, events


def valid_manifest():
    return dict(mode="development", models=["qwen"], selection_policy="decision_entity_link_v1",
                entity_reserve_fraction=.25, student_rollout_tokens=8,
                self_study_prompt_comparison="original_vs_study_only",
                holdout_evaluated_at_selection=False)


def test_runner_preserves_real_preflight_and_restores_all_hooks():
    impl, runtime, controls, old_method, old_selector, old_controls, physical, events = make_runner()
    with study_pair_runner(impl, runtime, controls, root="new", selector="linked", physical_controls=physical):
        assert impl.ROOT == "new" and runtime.FrozenJSONSelector == "linked"
        assert controls.original_question_controls is nullcontext
        assert impl.validate_manifest(valid_manifest()) == (200, ["qwen"])
        assert impl.preflight(None)["passed"]
        config = impl.replace(TTTConfig(), max_context=65536)
        assert (config.steps, config.max_answer_steps, config.value_learning_rate) == (2, 8, .0001)
        assert not config.align_student_teacher_prompt and config.standalone_answer_format == "original"
        assert runtime.MaskedTTT is not old_method and controls.original_question_controls is nullcontext
    assert impl.ROOT == "old" and runtime.FrozenJSONSelector is old_selector
    assert runtime.MaskedTTT is old_method and controls.original_question_controls is old_controls
    assert ("manifest", "cached_decision_score10") in events and ("preflight", True) in events


def test_runner_fails_closed_for_changed_protocol_and_restores_on_error():
    impl, runtime, controls, old_method, old_selector, old_controls, physical, _ = make_runner()
    with pytest.raises(ValueError, match="protocol changed"):
        with study_pair_runner(impl, runtime, controls, root="new", selector="linked", physical_controls=physical):
            impl.validate_manifest(dict(valid_manifest(), self_study_prompt_comparison="different"))
    assert impl.ROOT == "old" and runtime.FrozenJSONSelector is old_selector
    assert runtime.MaskedTTT is old_method and controls.original_question_controls is old_controls


def test_driver_adapter_runs_two_real_small_model_arms_and_rejects_cpu_as_gpu(tmp_path):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine()
    impl = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x, validate=lambda *_: None,
                           replace=replace, preflight=lambda _: {"passed": True})
    with study_pair_runner(impl, real_runtime, real_controls, root="candidate",
                           selector=LinkedFrozenSelector, physical_controls=kv_only_original_question_controls):
        engine.config = impl.replace(engine.config, max_context=engine.config.max_context,
            keep_fraction=.1, student_rollout_tokens=8, self_question_count=1, self_question_max_tokens=8,
            guard_enabled=False, selected_transfer_weight=0., value_basis_source="document",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        selector = real_runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with real_controls.original_question_controls():
            result = real_runtime.MaskedTTT(engine, selector).answer(
                Request(document=document, question=question, max_new_tokens=3))
        pair = result["study_prompt_pair"]
        assert pair["same_selection"] and pair["same_teacher_tokens"] and pair["same_zero_tokens"]
        assert pair["baseline_standalone_released"] and pair["candidate_standalone_released"]
        assert not pair["same_gpu"]  # CPU tests cannot masquerade as a real-GPU result.
        with pytest.raises(ValueError, match="physically released"):
            impl.validate(result, {"key": "case"})
