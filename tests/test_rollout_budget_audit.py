import json

import pytest

from experiments.anchored_short_verifier.audit_rollout_budget import audit, read_row, summarize


def _result(tokens=(1, 2), cap=2, complete=True):
    return {
        "training": {"student_rollout_limit": cap, "steps": [
            {"question_path": "main", "student_rollout_tokens": list(tokens)}
        ]},
        "teacher": {"complete": complete},
    }


def _saved(key, zero, trained):
    return {"case": key, "task": "qa1", "selective": zero, "cartridge": trained}


def test_reaching_cap_does_not_count_as_proven_truncation():
    row = read_row(_saved("x", 0, 1), _result())
    assert row["student_prefix_reaches_cap"] and row["student_prefix_tokens"] == 2
    assert "student_prefix_truncated" not in row
    assert read_row(_saved("x", 0, 1), _result(), eos_ids={2})["student_cap_without_eos"] is False
    assert read_row(_saved("x", 0, 1), _result(), eos_ids={3})["student_cap_without_eos"] is True
    assert read_row(_saved("x", 0, 1), _result())["student_cap_without_eos"] is None
    assert summarize([row])["delta_pp"] == 100
    with pytest.raises(ValueError):
        read_row(_saved("x", 0, 1), _result(tokens=(1, 2, 3), cap=2))
    with pytest.raises(ValueError):
        read_row(_saved("x", .2, 1), _result())


def test_only_fixed_snapshot_cases_are_aggregated(tmp_path):
    path = tmp_path / "cases/x/weight_0.00.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_result(complete=False)))
    snapshot = {
        "model": "test",
        "updated": 12,
        "rows": [_saved("x", 0, 1), _saved("missing", 1, 0)],
    }
    result = audit(snapshot, tmp_path)
    assert result["groups"]["all"]["n"] == 1
    assert result["groups"]["all"]["teacher_generation_incomplete"] == 1
    assert result["groups"]["shorter_than_prefix_cap"]["n"] == 0
    assert result["per_task"]["qa1"]["n"] == 1
    assert len(result["unavailable_cases"]) == 1
    assert result["new_model_calls"] == result["online_api_calls"] == 0
