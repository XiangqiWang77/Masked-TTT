from contextlib import nullcontext
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

import masked_ttt as real_runtime
import experiments.anchored_short_verifier.paired_controls as real_controls
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.rollout_value_pair import runner
from experiments.rollout_value_pair.runner import configs, pair_runner, real_pair_preflight
from masked_ttt import Request
from masked_ttt.config import TTTConfig
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_public_linked_selection import _source
from test_standalone import make_engine


def mock_driver():
    old_method, old_selector, old_controls, physical = object(), object(), object(), object()
    runtime = SimpleNamespace(MaskedTTT=old_method, FrozenJSONSelector=old_selector)
    controls = SimpleNamespace(original_question_controls=old_controls)
    def preflight(_):
        assert runtime.MaskedTTT is old_method and controls.original_question_controls is physical
        return dict(passed=True)
    impl = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x, validate=lambda *_: None,
                           replace=replace, preflight=preflight)
    return impl, runtime, controls, old_method, old_selector, old_controls, physical


def manifest():
    return dict(mode="development", models=["qwen"], selection_policy="decision_entity_link_v1",
                entity_reserve_fraction=.25, student_rollout_tokens=32, baseline_rollout_tokens=8,
                candidate_value_lr=.001, baseline_value_lr=.0001, holdout_evaluated_at_selection=False)


def test_config_changes_only_original_sd_hyperparameters():
    baseline, candidate = configs(TTTConfig())
    assert (baseline.student_rollout_tokens, baseline.value_learning_rate) == (8, .0001)
    assert (candidate.student_rollout_tokens, candidate.value_learning_rate) == (32, .001)
    assert {f.name for f in fields(baseline) if getattr(baseline, f.name) != getattr(candidate, f.name)} == {
        "student_rollout_tokens", "value_learning_rate"}
    assert candidate.selected_transfer_weight == 0 and candidate.value_basis_source == "document"


def test_runner_restores_hooks_and_checks_frozen_protocol(monkeypatch):
    impl, runtime, controls, method, selector, ctrl, physical = mock_driver()
    monkeypatch.setattr(runner, "real_pair_preflight", lambda _: dict(paired_rollout_value_preflight=True))
    with pair_runner(impl, runtime, controls, root="new", selector="linked", physical_controls=physical):
        assert impl.ROOT == "new" and runtime.FrozenJSONSelector == "linked"
        assert controls.original_question_controls is nullcontext
        assert impl.validate_manifest(manifest())["selection_policy"] == "cached_decision_score10"
        assert impl.preflight(None)["paired_rollout_value_preflight"]
        with pytest.raises(ValueError, match="protocol changed"):
            impl.validate_manifest(dict(manifest(), candidate_value_lr=.1))
    assert (impl.ROOT, runtime.MaskedTTT, runtime.FrozenJSONSelector, controls.original_question_controls) == (
        "old", method, selector, ctrl)


def test_small_model_pair_materializes_and_releases_both_arms(tmp_path):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine()
    impl = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x, validate=lambda *_: None,
                           replace=replace, preflight=lambda _: dict(passed=True))
    with pair_runner(impl, real_runtime, real_controls, root="candidate", selector=LinkedFrozenSelector,
                     physical_controls=kv_only_original_question_controls):
        engine.config = impl.replace(engine.config, max_context=engine.config.max_context, keep_fraction=.1,
            self_question_count=0, guard_enabled=False, selected_transfer_weight=0., value_basis_source="document",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        selector = real_runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with real_controls.original_question_controls():
            result = real_runtime.MaskedTTT(engine, selector).answer(
                Request(document=document, question=question, max_new_tokens=3))
        pair = result["rollout_value_pair"]
        assert pair["same_selection"] and pair["same_teacher_tokens"] and pair["same_zero_tokens"]
        assert pair["baseline_standalone_released"] and pair["candidate_standalone_released"]
        assert set(pair["changed_settings"]) == {"student_rollout_tokens", "value_learning_rate"}
        assert not pair["same_gpu"]  # CPU does not masquerade as the required GPU run.
        with pytest.raises(ValueError, match="hyperparameter arms changed"):
            impl.validate(result, {"key": "case"})


def test_real_gpu_preflight_cannot_claim_cpu_as_gpu():
    with pytest.raises(ValueError, match="allocated GPU"):
        real_pair_preflight(make_engine())
