from pathlib import Path
import pytest
from experiments.value_lr_physical import strong_dev_monitor


def report():
    return dict(mode="development", expected=200, errors=[], invalid=[],
        overall=dict(n=200, accuracy=dict(whole=50., zero=52., teacher=54., cartridge=56.)),
        versus_fixed_baseline=dict(n=180, delta_pp=1., new_correct=3, new_wrong=1))


def test_development_must_beat_same_run_teacher_zero_and_whole():
    value = report()
    assert strong_dev_monitor.development_decision(value)["eligible"]
    value["overall"]["accuracy"]["teacher"] = 56.
    assert not strong_dev_monitor.development_decision(value)["eligible"]
    value = report(); value["versus_fixed_baseline"].update(n=0, delta_pp=None)
    assert strong_dev_monitor.development_decision(value)["eligible"]
    value = report(); value["invalid"] = [{}]
    assert not strong_dev_monitor.development_decision(value)["eligible"]


def test_holdout_requests_one_gpu_and_short_backfill():
    script = strong_dev_monitor.holdout_script(Path("/tmp/query_cartridges_strong_value_gpt_holdout_20261104"))
    assert "#SBATCH --gpus-per-task=1" in script
    assert "#SBATCH --time-min=00:20:00" in script
    assert "value_lr_physical/strong_dev_execute.py" in script


def test_combine_checks_split_and_reports_both_parts(monkeypatch):
    monkeypatch.setattr(strong_dev_monitor, "validate_partition", lambda *_: None)
    def rows(prefix, n):
        return [dict(case=prefix + str(i), task="qa1", length=8192,
                     whole=0, zero=0, teacher=0, cartridge=1, linked8=None,
                     cartridge_percent=12.) for i in range(n)]
    dev_rows, held_rows = rows("dev", 200), rows("held", 800)
    manifest = dict(development_case_ids=[r["case"] for r in dev_rows],
                    holdout_case_ids=[r["case"] for r in held_rows])
    dev = dict(metric="answer_focused_lexical_v2", errors=[], invalid=[], rows=dev_rows,
               versus_fixed_baseline=dict(n=180, delta_pp=1.))
    held = dict(metric="answer_focused_lexical_v2", errors=[], invalid=[], rows=held_rows)
    result = strong_dev_monitor.combine(manifest, dev, held)
    assert result["overall"]["n"] == 1000 and result["holdout"]["n"] == 800
    held["rows"].pop()
    assert strong_dev_monitor.combine(manifest, dev, held) is None
    held["invalid"] = [{}]
    with pytest.raises(ValueError, match="invalid"):
        strong_dev_monitor.combine(manifest, dev, held)
