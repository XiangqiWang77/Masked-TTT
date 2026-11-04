import json
import subprocess
from types import SimpleNamespace

import pytest

import experiments.value_lr_physical.monitor_gain_transfer as monitor


def test_live_job_updates_hourly_state_without_restart(tmp_path, monkeypatch):
    root = tmp_path / "job"
    root.mkdir()
    (root / "submission.json").write_text(json.dumps(dict(job_id="123")))
    (root / "latest.json").write_text(json.dumps(dict(models=dict(
        qwen=dict(overall=dict(cases=4)), gpt=dict(overall=dict(cases=0))))))
    schedule = tmp_path / "schedule.json"
    schedule.write_text(json.dumps(dict(existing_key="preserved")))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="RUNNING|node\n")

    monkeypatch.setattr(monitor.subprocess, "run", run)
    result = monitor.monitor(root, schedule)
    assert result["state"] == "RUNNING" and result["complete"] == {"qwen": 4, "gpt": 0}
    assert len(calls) == 1 and calls[0][0] == "squeue"
    assert json.loads(schedule.read_text())["existing_key"] == "preserved"
    assert json.loads(root.joinpath("monitor.json").read_text())["job_id"] == "123"


@pytest.mark.parametrize("confirmed,expected", [("COMPLETED", "COMPLETED"),
                                                ("CANCELLED", "CANCELLED"), (None, "UNKNOWN")])
def test_missing_queue_record_requires_confirmed_terminal_state(confirmed, expected, monkeypatch):
    monkeypatch.setattr(monitor.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    monkeypatch.setattr(monitor, "confirmed_terminal", lambda _job: confirmed)
    state, reason = monitor.scheduler_state("123")
    assert state == expected and reason is None


def test_scheduler_error_is_not_misreported_as_terminal(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "squeue")

    monkeypatch.setattr(monitor.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        monitor.scheduler_state("123")
