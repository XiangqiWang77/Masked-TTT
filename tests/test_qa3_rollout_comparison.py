import copy
import json

import pytest

from experiments.anchored_short_verifier.audit_qa3_rollout_comparison import audit, compare_pair


CASE = "babilong-8-qa3-300060"


def record(limit, tokens, kl):
    return {"key": CASE, "task": "qa3", "model": "test",
        "selection": {"source_case_key": CASE, "selected_original_chunks": [[2], [5]]},
        "teacher": {"tokens": [7]}, "self_study": {"checks": []},
        "training": {"student_rollout_limit": limit, "selected_transfer_weight": 0,
            "steps": [{"question_path": "main", "student_rollout_tokens": tokens,
                       "target_positions": list(range(len(tokens))), "teacher_KL_before": kl,
                       "gain_gradient_norm": kl + 0.1, "value_gradient_norm": kl + 0.2}]},
        "standalone_cartridge": {}}


def records():
    old, new = record(8, [1, 2], 0.001), record(32, [1, 2, 3], 0.5)
    old["standalone_cartridge"]["paired_prompt_control"] = {
        "zero_original": {"tokens": [8]}, "trained_original": {"tokens": [8]},
        "reference_answers_used": False, "teacher_answer_in_student_prompt": False}
    new["standalone_cartridge"]["original_question_controls"] = {
        "zero": {"tokens": [8]}, "adapted": {"tokens": [9]},
        "reference_answers_used": False, "teacher_answer_in_student_prompt": False}
    return old, new


def decode(tokens):
    return "The milk was in the garden" if len(tokens) > 2 else "The milk was in the"


def test_compares_same_evidence_prefix_and_full_causal_targets():
    old, new = records()
    row = compare_pair(CASE, old, new, decode)
    assert row["longer"] and row["new_has_predicated_location"]
    assert not row["old_has_predicated_location"] and row["final_cartridge_changed"]
    assert row["old_first_kl"] == .001 and row["new_first_kl"] == .5
    for field in ("teacher", "selection", "prefix", "targets"):
        broken = copy.deepcopy(new)
        if field == "teacher": broken["teacher"]["tokens"] = [3]
        elif field == "selection": broken["selection"]["selected_original_chunks"] = [[1]]
        elif field == "prefix": broken["training"]["steps"][0]["student_rollout_tokens"][0] = 8
        else: broken["training"]["steps"][0]["target_positions"] = [0, 2]
        with pytest.raises(ValueError): compare_pair(CASE, old, broken, decode)


def test_audit_reports_only_available_fixed_development_pairs(tmp_path):
    old, new = records()
    old_root, other_root, new_root = (tmp_path / name for name in ("old", "other", "new"))
    for path, value in ((old_root / "cases" / CASE / "weight_0.00.json", old),
                        (new_root / "cases" / "qwen" / CASE / "result.json", new)):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    missing = "babilong-16-qa3-300060"
    result = audit([CASE, missing], "qwen", old_root, other_root, new_root, decode)
    assert result["qa3_expected"] == 2 and result["matched"] == 1
    assert result["longer"] == result["new_with_location"] == 1
    assert result["old_with_location"] == 0 and len(result["missing"]) == 1
    assert result["new_median_first_kl"] == .5 and not result["references_read"]
