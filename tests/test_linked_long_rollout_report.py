import copy
import json

import pytest

from experiments.babilong_linked_selection.long_rollout_protocol import summarize
from experiments.babilong_linked_selection.long_rollout_report import combine_reports, collect, render, validate_new


def fixture_result(case):
    selection = {"source_case_key": case, "backend": "precomputed_source_JSON",
        "selection_policy": "decision_entity_link_v1", "entity_reserve_fraction": .25,
        "total_chunks": 10, "selected_chunks": [2], "selected_original_chunks": [[2]],
        "original_prompt_tokens": 100, "complete_original_document": True,
        "reference_answers_used": False, "online_api_calls": 0}
    training = {"objective": "on-policy KL(teacher || student)", "value_basis_source": "document",
        "selected_transfer_weight": 0, "student_rollout_limit": 32,
        "teacher_conditioned_on_student_prefix": True, "student_rollouts_refreshed_each_step": True,
        "optimizer": "Adam", "teacher_cap": 32, "teacher_target_scope": "choice_or_all",
        "adaptive_max_answer_steps": 8, "adaptive_answer_KL_stop": .5,
        "steps": [{"question_path": "main", "student_rollout_tokens": [1, 2, 3]}]}
    controls = {"zero": {"tokens": [4], "prediction": "no"},
        "adapted": {"tokens": [9], "prediction": "yes"}, "reference_answers_used": False,
        "teacher_answer_in_student_prompt": False, "same_selected_kv": True,
        "same_question_suffix": True, "same_generation_limit": True, "teacher_suffix_matches_student": False}
    return {"key": case, "model": "Qwen/Qwen3-4B-Instruct-2507", "task": "qa3", "length": 8192,
        "keep_fraction": .1, "selection": selection, "training": training, "prediction": "yes",
        "output_source": "standalone_cartridge", "generated_token_ids": [9],
        "teacher": {"tokens": [7], "prediction": "yes", "response_instruction_applied": True},
        "self_study": {"checks": [], "reference_answers_used": False, "student_answer_hint_used": False},
        "self_judge": {"native_prediction": "no"}, "backbone_unchanged": True,
        "original_document_KV_unchanged": True, "teacher_answer_in_final_prompt": False,
        "memory": {"full_document_kv_bytes": 1000, "full_cache_release_before_standalone":
                   {"requested": True, "full_cache_tensors_live_before_standalone": 0}},
        "standalone_cartridge": {"teacher_answer_in_final_prompt": False, "prediction": "yes", "generated_token_ids": [9],
            "original_question_controls": controls, "memory": {"standalone_cartridge_bytes": 200,
                "kv_only_materialized": True, "cartridge_adapter_bytes": 0,
                "additional_source_kv_bytes_before_decode": 0,
                "additional_training_adapter_bytes_before_decode": 0}}}


def fixture_manifest():
    lengths = (8, 16, 32, 64)
    dev = [f"babilong-{length}-qa3-{300100 + index}" for index in range(50) for length in lengths]
    sources = ["qa1-300023", "qa2-300037"] + [f"qa4-{310000 + index}" for index in range(198)]
    hold = [f"babilong-{length}-{source}" for source in sources for length in lengths]
    return {"case_ids": dev, "development_case_ids": dev, "holdout_case_ids": hold,
        "case_count": 200, "mode": "development", "models": ["qwen"],
        "selection_policy": "decision_entity_link_v1", "entity_reserve_fraction": .25,
        "keep_fraction": .1, "student_rollout_tokens": 32, "selected_transfer_weight": 0,
        "reference_answers_used_in_inference": False, "online_api_calls": 0,
        "qualitative_case_sources": ["qa1-300023", "qa2-300037"]}


def test_report_requires_the_original_full_student_and_released_final_state():
    case = "babilong-8-qa3-300100"
    result = fixture_result(case)
    validate_new(case, result, "qwen", "qa3")
    for kind in ("selected_training", "retained_cache", "leaked_answer", "output_mismatch", "full_output"):
        bad = copy.deepcopy(result)
        if kind == "selected_training": bad["training"]["selected_transfer_weight"] = .25
        elif kind == "retained_cache": bad["standalone_cartridge"]["memory"]["additional_source_kv_bytes_before_decode"] = 5
        elif kind == "leaked_answer": bad["teacher_answer_in_final_prompt"] = True
        elif kind == "output_mismatch": bad["standalone_cartridge"]["generated_token_ids"] = [8]
        else: bad["output_source"] = "full_context"
        with pytest.raises(ValueError): validate_new(case, bad, "qwen", "qa3")


def test_partial_report_scores_only_generated_cases_and_keeps_matched_baseline(tmp_path):
    manifest = fixture_manifest()
    case = manifest["case_ids"][0]
    new = fixture_result(case)
    old = copy.deepcopy(new)
    old["training"]["student_rollout_limit"] = 8
    old["training"]["steps"][0]["student_rollout_tokens"] = [1, 2]
    old["standalone_cartridge"]["paired_prompt_control"] = {
        "zero_original": {"tokens": [4]}, "trained_original": {"prediction": "no"},
        "same_selected_kv": True, "same_generation_limit": True,
        "reference_answers_used": False, "teacher_answer_in_student_prompt": False}
    candidate, baseline = tmp_path / "candidate", tmp_path / "baseline"
    for path, value in ((candidate / "cases" / "qwen" / case / "result.json", new),
                        (baseline / "cases" / "qwen" / case / "result.json", old)):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    refs = {case: {"task": "qa3", "answers": ["yes"]}}
    score = lambda prediction, answers, _benchmark, _task: (int(prediction in answers), True)
    report = collect(manifest, candidate, [baseline], refs, score)
    overall = report["models"]["qwen"]["overall"]
    assert overall["n"] == 1 and overall["accuracy"]["teacher"] == 100
    assert overall["versus_linked8"]["n"] == 1 and overall["versus_linked8"]["delta_pp"] == 100
    assert overall["median_cartridge_percent"] == 20
    assert report["models"]["qwen"]["decision"]["reason"] == "development_incomplete"
    assert report["evaluation_answers_used_only_posthoc"] and report["online_api_calls"] == 0
    text = render(report)
    assert "| qwen | 1 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 | 20.00 |" in text
    assert "| qwen | 1 | 0.00 | 100.00 | 1 / 0 | 100.00 |" in text


def test_combined_report_requires_complete_source_disjoint_200_and_800():
    manifest = fixture_manifest()
    def rows(cases, teacher_right):
        return [dict(case=case, task=case.split("-")[2], length=int(case.split("-")[1]) * 1024,
                     whole=0, zero=0, teacher=int(index < teacher_right), cartridge=1,
                     linked8=0, cartridge_percent=20)
                for index, case in enumerate(cases)]
    dev_rows, held_rows = rows(manifest["case_ids"], 0), rows(manifest["holdout_case_ids"], 200)
    common = {"metric": "answer_focused_lexical_v2", "selection_policy": "decision_entity_link_v1",
        "student_rollout_tokens": 32, "selected_transfer_weight": 0, "cached_scores": True,
        "online_api_calls": 0, "evaluation_answers_used_only_posthoc": True, "errors": [], "invalid": []}
    dev = dict(common, mode="development", expected_per_model=200,
               models={"qwen": {"rows": dev_rows, "overall": summarize(dev_rows)}})
    held = dict(common, mode="holdout", expected_per_model=800,
                models={"qwen": {"rows": held_rows, "overall": summarize(held_rows),
                    "decision": {"reason": "holdout_target_met"},
                    "excluding_viewed_examples": summarize(held_rows)}})
    combined = combine_reports(dev, held)
    total = combined["models"]["qwen"]["overall"]
    assert total["n"] == 1000 and total["accuracy"]["teacher"] == 20
    assert total["accuracy"]["cartridge"] == 100 and "1,000 cases" in render(combined)
    partial = copy.deepcopy(held)
    partial["models"]["qwen"]["rows"].pop()
    assert combine_reports(dev, partial) is None
    overlap = copy.deepcopy(held)
    overlap["models"]["qwen"]["rows"][0]["case"] = dev_rows[0]["case"]
    with pytest.raises(ValueError): combine_reports(dev, overlap)
