import json

import pytest

from experiments.value_lr_physical.recovery_report import collect, render, summarize
from experiments.value_lr_physical.recovery_watch import advance


def test_recovery_and_retention_count_paired_answers():
    rows = [dict(whole=1, zero=0, teacher=0, cartridge=1),
            dict(whole=1, zero=0, teacher=1, cartridge=0),
            dict(whole=0, zero=1, teacher=1, cartridge=1),
            dict(whole=0, zero=1, teacher=0, cartridge=0)]
    got = summarize(rows)
    assert got["whole_only"] == dict(total=2, recovered=1, percent=50)
    assert got["selected_only"] == dict(total=2, retained=1, percent=50)
    assert got["versus_zero"] == dict(new_correct=1, new_wrong=1, delta_points=0)
    assert got["versus_teacher"] == dict(new_correct=1, new_wrong=1, delta_points=0)


def test_report_validates_coverage_and_does_not_turn_missing_cases_into_errors():
    row = dict(case="babilong-8-qa1-300000", task="qa1", whole=1, zero=0, teacher=0, cartridge=1)
    report = dict(mode="holdout", model="gpt", expected=800, metric="answer_focused_lexical_v2",
                  fixed_value_learning_rate=.001, overall=dict(n=1), rows=[row], errors=[], invalid=[])
    result = collect(report)
    assert result["overall"]["n"] == 1 and result["expected"] == 800
    assert result["overall"]["whole_only"]["recovered"] == 1
    assert "1 / 800 paired cases" in render(result)
    report["rows"].append(row)
    report["overall"]["n"] = 2
    with pytest.raises(ValueError, match="paired"):
        collect(report)


def test_empty_and_nonbinary_groups_are_explicit():
    assert summarize([])["accuracy"]["whole"] is None
    assert summarize([])["whole_only"]["percent"] is None
    with pytest.raises(ValueError, match="binary"):
        summarize([dict(whole=2, zero=0, teacher=0, cartridge=1)])


def test_watcher_uses_only_existing_reports_and_waits_for_both_models(tmp_path):
    root = tmp_path / "query_cartridges_value_lr_physical_gpt_holdout_20261104"
    root.mkdir()
    row = dict(case="babilong-8-qa1-300000", task="qa1", whole=1, zero=0, teacher=0, cartridge=1)
    report = dict(mode="holdout", model="gpt", expected=800, metric="answer_focused_lexical_v2",
                  fixed_value_learning_rate=.001, overall=dict(n=1), rows=[row], errors=[], invalid=[])
    (root / "latest.json").write_text(json.dumps(report))
    status = advance(tmp_path)
    assert status["stage"] == "waiting" and status["models"]["gpt"]["n"] == 1
    assert status["models"]["qwen"]["stage"] == "waiting_for_report"
    assert "1 / 800 paired cases" in (root / "recovery_latest.md").read_text()
    (root / "monitor_status.json").write_text(json.dumps(dict(stage="terminal_without_complete_results")))
    (tmp_path / "query_cartridges_value_lr_physical_qwen_launcher_status.json").write_text(
        json.dumps(dict(stage="no_eligible_model")))
    assert advance(tmp_path)["stage"] == "all_terminal"
