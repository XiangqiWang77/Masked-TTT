import copy

import pytest

from experiments.babilong_linked_selection.long_rollout_protocol import (
    TEACHER_INSTRUCTION, development_decision, holdout_decision, prepare_development_manifest,
    prepare_holdout_manifest, summarize, validate_linked_baseline, validate_partition,
)


def pair():
    training = {"objective": "on-policy KL(teacher || student)", "optimizer": "Adam",
        "teacher_cap": 32, "teacher_target_scope": "choice_or_all", "adaptive_max_answer_steps": 8,
        "adaptive_answer_KL_stop": .5, "selected_transfer_weight": 0, "value_basis_source": "document"}
    shared = {"key": "x", "task": "qa3", "model": "qwen",
        "selection": {"source_case_key": "x", "selection_policy": "decision_entity_link_v1",
                      "entity_reserve_fraction": .25, "selected_original_chunks": [[2]], "total_chunks": 10,
                      "original_prompt_tokens": 100, "complete_original_document": True,
                      "reference_answers_used": False, "online_api_calls": 0},
        "teacher": {"tokens": [7], "response_instruction_applied": True}, "self_study": {"checks": []}}
    old, new = copy.deepcopy(shared), copy.deepcopy(shared)
    old["training"] = dict(training, student_rollout_limit=8,
                           steps=[{"question_path": "main", "student_rollout_tokens": [1, 2]}])
    new["training"] = dict(training, student_rollout_limit=32,
                           steps=[{"question_path": "main", "student_rollout_tokens": [1, 2, 3]}])
    old["standalone_cartridge"] = {"paired_prompt_control": {"zero_original": {"tokens": [4]},
        "same_selected_kv": True, "same_generation_limit": True,
        "reference_answers_used": False, "teacher_answer_in_student_prompt": False}}
    new["standalone_cartridge"] = {"original_question_controls": {"zero": {"tokens": [4]},
        "same_selected_kv": True, "same_generation_limit": True, "same_question_suffix": True,
        "reference_answers_used": False, "teacher_answer_in_student_prompt": False}}
    return old, new


def test_strict_pair_rejects_changed_teacher_evidence_and_student_prefix():
    old, new = pair()
    validate_linked_baseline("x", old, new)
    for kind in ("teacher", "evidence", "prefix", "zero", "objective"):
        bad = copy.deepcopy(new)
        if kind == "teacher": bad["teacher"]["tokens"] = [5]
        elif kind == "evidence": bad["selection"]["selected_original_chunks"] = [[1]]
        elif kind == "prefix": bad["training"]["steps"][0]["student_rollout_tokens"] = [8, 2]
        elif kind == "zero": bad["standalone_cartridge"]["original_question_controls"]["zero"]["tokens"] = [5]
        else: bad["training"]["selected_transfer_weight"] = .1
        with pytest.raises(ValueError): validate_linked_baseline("x", old, bad)


def test_development_and_holdout_require_teacher_and_paired_improvement():
    rows = [dict(whole=0, zero=0, teacher=0, cartridge=1, linked8=0) for _ in range(4)]
    summary = summarize(rows)
    assert development_decision(summary, required=4, minimum_matched=4)["eligible"]
    assert holdout_decision(summary, required=4, minimum_matched=4)["eligible"]
    rows[0]["linked8"] = None
    assert development_decision(summarize(rows), required=4, minimum_matched=4)["reason"] == \
        "development_or_matched_baseline_incomplete"
    assert not development_decision(summarize([dict(whole=0, zero=0, teacher=1, cartridge=1, linked8=0)]),
                                    required=1, minimum_matched=1)["eligible"]


def test_partition_is_disjoint_by_source_not_only_by_case():
    lengths = (8, 16, 32, 64)
    development = [f"babilong-{length}-qa3-{300100 + index}"
                   for index in range(50) for length in lengths]
    sources = ["qa1-300023", "qa2-300037"] + [f"qa4-{310000 + index}" for index in range(198)]
    holdout = [f"babilong-{length}-{source}" for source in sources for length in lengths]
    validate_partition(development, holdout)
    overlap = [case for case in holdout if not case.endswith("qa4-310000")]
    overlap += [f"babilong-{length}-qa3-300100" for length in lengths]
    with pytest.raises(ValueError): validate_partition(development, overlap)
    with pytest.raises(ValueError): validate_partition(development, holdout[:-1])


def test_promotion_keeps_fixed_split_and_only_selects_winning_models():
    lengths = (8, 16, 32, 64)
    dev = [f"babilong-{length}-qa3-{300100 + i}" for i in range(50) for length in lengths]
    sources = ["qa1-300023", "qa2-300037"] + [f"qa4-{310000 + i}" for i in range(198)]
    hold = [f"babilong-{length}-{source}" for source in sources for length in lengths]
    common = {"case_ids": dev, "holdout_case_ids": hold, "model_protocols": {"qwen": "q", "gpt": "g"},
        "mode": "development", "case_count": 200, "models": ["qwen", "gpt"], "keep_fraction": .1,
        "basis_source": "document", "selected_transfer_weight": 0,
        "reference_answers_used_in_inference": False, "online_api_calls": 0}
    rollout = dict(common, selection_policy="cached_decision_score10", student_rollout_tokens=32,
                   teacher_instruction=TEACHER_INSTRUCTION, verifier_policy="original_exact")
    linked = dict(common, selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25)
    profile = {"teacher_response_instruction": TEACHER_INSTRUCTION, "student_rollout_tokens": 8,
               "steps": 2, "gain_learning_rate": .01, "value_learning_rate": .0001}
    manifest = prepare_development_manifest(rollout, linked, profile)
    assert manifest["case_ids"] == dev and manifest["development_case_ids"] == dev
    assert manifest["selection_policy"] == "decision_entity_link_v1"
    winning_rows = [dict(case=case, whole=0, zero=0, teacher=0, cartridge=1, linked8=0) for case in dev]
    tied_rows = [dict(case=case, whole=0, zero=0, teacher=1, cartridge=1, linked8=0) for case in dev]
    winning, tied = summarize(winning_rows), summarize(tied_rows)
    report = {"mode": "development", "expected_per_model": 200, "metric": "answer_focused_lexical_v2",
        "selection_policy": "decision_entity_link_v1", "cached_scores": True,
        "student_rollout_tokens": 32, "selected_transfer_weight": 0,
        "evaluation_answers_used_only_posthoc": True, "online_api_calls": 0, "errors": [], "invalid": [],
        "models": {"qwen": {"overall": winning, "rows": winning_rows, "decision": development_decision(winning)},
                   "gpt": {"overall": tied, "rows": tied_rows, "decision": development_decision(tied)}}}
    promoted = prepare_holdout_manifest(manifest, report)
    assert promoted["case_ids"] == hold and promoted["models"] == ["qwen"]
    assert promoted["mode"] == "holdout" and promoted["holdout_evaluated_at_selection"] is False
    with pytest.raises(ValueError):
        prepare_development_manifest(rollout, dict(linked, entity_reserve_fraction=.5), profile)
    report["models"]["qwen"]["decision"]["eligible"] = False
    with pytest.raises(ValueError): prepare_holdout_manifest(manifest, report)
