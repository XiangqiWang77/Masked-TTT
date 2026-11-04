from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

import masked_ttt as real_runtime
import experiments.anchored_short_verifier.paired_controls as real_controls
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.value_lr_physical import canonical_runner
from experiments.value_lr_physical.canonical_runner import canonical_pair_runner, real_pair_preflight
from masked_ttt import Request
from masked_ttt.config import TTTConfig
from test_canonical_pair import BoundaryTokenizer, Selector
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
        assert runtime.MaskedTTT is old_method and controls.original_question_controls is physical_controls
        events.append(("preflight", True))
        return {"passed": True}

    implementation = SimpleNamespace(ROOT="old", validate_manifest=manifest, validate=lambda *args: None,
                                     replace=replace, preflight=preflight)
    return implementation, runtime, controls, old_method, old_selector, old_controls, physical_controls, events


def valid_manifest():
    return dict(mode="development", models=["qwen", "gpt"], selection_policy="decision_entity_link_v1",
                entity_reserve_fraction=.25, student_rollout_tokens=8,
                evidence_tokenization_comparison="original_vs_canonical_fragment",
                holdout_evaluated_at_selection=False)


def test_hooks_restore_and_older_preflight_runs_unmodified(monkeypatch):
    impl, runtime, controls, old_method, old_selector, old_controls, physical, events = make_runner()
    monkeypatch.setattr(canonical_runner, "real_pair_preflight", lambda _: dict(canonical_paired_preflight=True))
    with canonical_pair_runner(impl, runtime, controls, root="new", selector="linked", physical_controls=physical):
        assert impl.ROOT == "new" and runtime.FrozenJSONSelector == "linked"
        assert controls.original_question_controls is nullcontext
        assert impl.validate_manifest(valid_manifest()) == (200, ["qwen", "gpt"])
        assert impl.preflight(None) == dict(passed=True, canonical_paired_preflight=True)
        config = impl.replace(TTTConfig(), max_context=65536)
        assert (config.steps, config.max_answer_steps, config.value_learning_rate) == (2, 8, .0001)
        assert not config.align_student_teacher_prompt and config.standalone_answer_format == "original"
        assert runtime.MaskedTTT is not old_method and controls.original_question_controls is nullcontext
    assert impl.ROOT == "old" and runtime.FrozenJSONSelector is old_selector
    assert runtime.MaskedTTT is old_method and controls.original_question_controls is old_controls
    assert ("manifest", "cached_decision_score10") in events and ("preflight", True) in events


def test_changed_protocol_fails_closed_and_restores():
    impl, runtime, controls, old_method, old_selector, old_controls, physical, _ = make_runner()
    with pytest.raises(ValueError, match="protocol changed"):
        with canonical_pair_runner(impl, runtime, controls, root="new", selector="linked", physical_controls=physical):
            impl.validate_manifest(dict(valid_manifest(), evidence_tokenization_comparison="different"))
    assert impl.ROOT == "old" and runtime.FrozenJSONSelector is old_selector
    assert runtime.MaskedTTT is old_method and controls.original_question_controls is old_controls


def test_driver_adapter_runs_real_cpu_pair_without_claiming_gpu():
    engine = make_engine()
    engine.tokenizer = BoundaryTokenizer()
    impl = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x, validate=lambda *_: None,
                           replace=replace, preflight=lambda _: {"passed": True})
    with canonical_pair_runner(impl, real_runtime, real_controls, root="candidate",
                               selector=Selector, physical_controls=kv_only_original_question_controls):
        engine.config = impl.replace(engine.config, max_context=engine.config.max_context,
            chunk_tokens=32, self_question_count=0, guard_enabled=True, selected_transfer_weight=0.,
            value_basis_source="document", release_full_before_cartridge=True,
            release_selected_source_before_decode=True)
        with real_controls.original_question_controls():
            result = real_runtime.MaskedTTT(engine, Selector()).answer(
                Request(document="A" * 32 + "B" * 32 + "E" * 32 + "F" * 32,
                        question="Which block is E?", max_new_tokens=3))
        pair = result["canonical_tokenization_pair"]
        assert pair["same_selection"] and pair["baseline_standalone_released"]
        assert pair["candidate_standalone_released"] and not pair["same_gpu"]
        assert pair["baseline_teacher_prompt_tokens"] == pair["candidate_teacher_prompt_tokens"] + 2
        with pytest.raises(ValueError, match="physically released"):
            impl.validate(result, {"key": "case"})


def test_real_gpu_preflight_cannot_claim_success_on_cpu():
    with pytest.raises(ValueError, match="allocated GPU"):
        real_pair_preflight(make_engine())


def test_gpu_preflight_restores_config_after_failure(monkeypatch):
    engine = SimpleNamespace(device=SimpleNamespace(type="cuda"), config=TTTConfig())
    original = engine.config
    def fail(*_args, **_kwargs):
        assert engine.config.self_question_count == 0 and engine.config.guard_enabled
        raise RuntimeError("synthetic preflight failure")
    monkeypatch.setattr(canonical_runner, "run_pair", fail)
    with pytest.raises(RuntimeError, match="synthetic preflight failure"):
        real_pair_preflight(engine)
    assert engine.config is original
