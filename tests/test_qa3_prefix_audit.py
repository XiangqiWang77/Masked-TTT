import json

import pytest

from experiments.anchored_short_verifier.audit_qa3_prefix import audit, audit_result


def record(case, tokens=(1, 2), limit=2, target_positions=(0, 1)):
    return {"key": case, "task": "qa3", "training": {"student_rollout_limit": limit,
        "steps": [{"question_path": "main", "student_rollout_tokens": list(tokens),
                   "target_positions": list(target_positions)}]}}


def test_end_of_location_slot_is_not_a_final_location():
    case = "babilong-8-qa3-300060"
    unfinished = audit_result(case, record(case), lambda _: "Before the kitchen the milk was in the")
    assert unfinished["reached_cap"] and unfinished["ends_at_location_slot"]
    finished = audit_result(case, record(case), lambda _: "Before the kitchen the milk was in the garden")
    assert not finished["ends_at_location_slot"]
    with pytest.raises(ValueError):
        audit_result(case, record(case, target_positions=(0, 2)), lambda _: "")


def test_audit_only_uses_manifest_cases_and_reports_missing(tmp_path):
    case, missing = "babilong-8-qa3-300060", "babilong-16-qa3-300060"
    path = tmp_path / "cases" / case / "weight_0.00.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record(case)))
    result = audit([case, missing, "babilong-8-qa1-300000"], tmp_path,
                   lambda _: "The milk was in the")
    assert result["cases_expected"] == 2 and result["cases_observed"] == 1
    assert result["both"] == 1 and result["source_questions"] == 1
    assert result["missing_cases"] == [missing]
    assert not result["references_read"] and result["new_model_calls"] == 0
    with pytest.raises(ValueError):
        audit([case, case], tmp_path, lambda _: "")
