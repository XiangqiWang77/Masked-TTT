import copy
import json

import pytest

from experiments.babilong_linked_selection.audit_qa3_teacher import MODEL, audit, validate


def result(case):
    return {"key": case, "task": "qa3", "model": MODEL,
            "selection": {"source_case_key": case, "reference_answers_used": False, "online_api_calls": 0,
                "complete_original_document": True, "selected_chunks": [2], "total_chunks": 10,
                "original_prompt_tokens": 100, "selection_policy": "decision_entity_link_v1",
                "entity_reserve_fraction": .25},
            "teacher": {"response_instruction_applied": True, "prediction": "yes"}}


def test_teacher_audit_rejects_different_source_or_budget():
    old, new = result("x"), result("x")
    validate("x", old, new)
    bad = copy.deepcopy(new)
    bad["selection"]["total_chunks"] = 20
    with pytest.raises(ValueError): validate("x", old, bad)
    bad = copy.deepcopy(new)
    bad["selection"]["reference_answers_used"] = True
    with pytest.raises(ValueError): validate("x", old, bad)


def test_teacher_audit_scores_only_completed_development_cases(tmp_path):
    case = "babilong-8-qa3-300060"
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old, new = result(case), result(case)
    old["teacher"]["prediction"] = "no"
    for path, value in ((old_root / "cases" / case / "weight_0.00.json", old),
                        (new_root / "cases/qwen" / case / "result.json", new)):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    score = lambda text, answers, _benchmark, _task: (int(text in answers), True)
    output = audit([case], old_root, new_root, {case: {"task": "qa3", "answers": ["yes"]}}, score)
    assert output["overall"]["score_only_correct"] == 0 and output["overall"]["linked_correct"] == 1
    assert output["overall"]["new_correct"] == 1 and output["source_questions"] == 1
    assert output["reference_answers_used_only_posthoc"] and output["online_api_calls"] == 0
