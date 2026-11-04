"""The query-head variant changes only the selected gain parameterization."""
from dataclasses import replace
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import masked_ttt as runtime
import experiments.anchored_short_verifier.paired_controls as controls
from experiments.gqa_head_gain.runner import query_head_runner
from experiments.gqa_head_gain.report import collect, validate_candidate, validate_pair
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from masked_ttt import MaskedTTT, Request, TTTConfig
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_public_linked_selection import _source
from test_standalone import make_engine


def test_runner_restores_hooks_on_error_and_checks_frozen_protocol():
    implementation = SimpleNamespace(ROOT="original", validate_manifest=lambda x: x,
                                     validate=lambda _result, _entry: None, replace=replace)
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.replace, runtime.FrozenJSONSelector, controls.original_question_controls)
    with pytest.raises(RuntimeError, match="intentional"):
        with query_head_runner(implementation, runtime, controls, root="candidate",
                               linked_selector=LinkedFrozenSelector,
                               physical_controls=kv_only_original_question_controls):
            manifest = dict(gain_head_mode="query", selection_policy="decision_entity_link_v1",
                            entity_reserve_fraction=.25, student_rollout_tokens=8)
            assert implementation.validate_manifest(manifest)["selection_policy"] == "cached_decision_score10"
            assert implementation.replace(TTTConfig(), steps=3).gain_head_mode == "query"
            with pytest.raises(ValueError):
                implementation.validate_manifest(dict(manifest, student_rollout_tokens=32))
            raise RuntimeError("intentional")
    assert (implementation.ROOT, implementation.validate_manifest, implementation.validate,
            implementation.replace, runtime.FrozenJSONSelector, controls.original_question_controls) == old


def test_runner_uses_linked_frozen_scores_and_independent_final_kv(tmp_path):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine()
    engine.eos_ids = {-1}
    implementation = SimpleNamespace(ROOT="original", validate_manifest=lambda x: x,
                                     validate=lambda _result, _entry: None, replace=replace)
    with query_head_runner(implementation, runtime, controls, root="candidate",
                           linked_selector=LinkedFrozenSelector,
                           physical_controls=kv_only_original_question_controls):
        engine.config = implementation.replace(engine.config, keep_fraction=.1, student_rollout_tokens=8,
            self_question_count=0, guard_enabled=False, selected_transfer_weight=0.,
            value_basis_source="document", release_full_before_cartridge=True,
            release_selected_source_before_decode=True,
            teacher_response_instruction="Return only the final answer.")
        selector = runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with controls.original_question_controls():
            result = MaskedTTT(engine, selector).answer(Request(
                document=document, question=question, max_new_tokens=8))
        implementation.validate(result, {"key": "case"})
    assert result["training"]["gain_head_mode"] == "query"
    assert result["training"]["objective"] == "on-policy KL(teacher || student)"
    assert len(result["training"]["steps"][0]["student_rollout_tokens"]) == 8
    assert result["training"]["steps"][0]["gain_gradient_norm"] > 0
    assert result["training"]["steps"][0]["value_gradient_norm"] > 0
    assert result["selection"]["selected_chunks"] == [1, 2, 3, 6]
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    result.update(key="case", model="qwen-test", model_short="qwen", task="qa1", length=8192,
                  keep_fraction=.1)
    result["self_judge"]["native_prediction"] = "blue"
    validate_candidate(result, "case", "qwen", "qa1")
    previous = deepcopy(result)
    old_controls = previous["standalone_cartridge"]["original_question_controls"]
    previous["standalone_cartridge"]["paired_prompt_control"] = dict(
        zero_original=old_controls["zero"], trained_original=old_controls["adapted"],
        same_selected_kv=True, same_generation_limit=True, reference_answers_used=False)
    validate_pair(previous, result, "case")
    unmatched = deepcopy(previous)
    unmatched["teacher"]["tokens"] = [999]
    with pytest.raises(ValueError):
        validate_pair(unmatched, result, "case")

    candidate = tmp_path / "candidate/cases/qwen/case/result.json"
    baseline = tmp_path / "baseline/cases/qwen/case/result.json"
    candidate.parent.mkdir(parents=True); baseline.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(result)); baseline.write_text(json.dumps(previous))
    manifest = dict(case_count=200, case_ids=["case"] + [f"missing-{i}" for i in range(199)],
        models=["qwen"], gain_head_mode="query", keep_fraction=.1, student_rollout_tokens=8,
        selected_transfer_weight=0., selection_policy="decision_entity_link_v1",
        reference_answers_used_in_inference=False, online_api_calls=0)
    refs = {"case": dict(task="qa1", answers=["blue"])}
    report = collect(manifest, tmp_path / "candidate", tmp_path / "baseline", refs,
                     lambda text, answers, benchmark, task: (int("blue" in text),))
    assert not report["errors"] and not report["invalid"]
    assert report["models"]["qwen"]["overall"]["n"] == 1
    assert report["models"]["qwen"]["overall"]["strictly_paired"]["n"] == 1
    assert not report["models"]["qwen"]["development_gate"]["eligible"]
