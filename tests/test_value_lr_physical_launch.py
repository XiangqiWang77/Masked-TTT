import json
from types import SimpleNamespace

import pytest

import experiments.value_lr_physical.launch_when_ready as launcher
from test_value_lr_physical import fixture_value_development


def make_development(tmp_path):
    development, report, _ = fixture_value_development()
    report["models"]["qwen"] = json.loads(json.dumps(report["models"]["gpt"]))
    root = tmp_path / "query_cartridges_value_lr_20261104"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(development))
    (root / "latest.json").write_text(json.dumps(report))
    return root, development, report


def test_no_submission_until_all_200_model_cases_are_scored(tmp_path, monkeypatch):
    root, _, report = make_development(tmp_path)
    report["models"]["qwen"]["overall"]["n"] = 199
    (root / "latest.json").write_text(json.dumps(report))
    monkeypatch.setattr(launcher, "prepare", lambda *_args: pytest.fail("early preparation"))
    assert launcher.advance(tmp_path, tmp_path / "python", "qwen") == {
        "stage": "development_in_progress", "model": "qwen", "cases": 199}


def test_complete_model_submits_once_and_enables_hourly_reporting(tmp_path, monkeypatch):
    make_development(tmp_path)
    monkeypatch.setattr(launcher, "prepare", lambda *_args: {"cpu_tests_passed": True})
    monkeypatch.setattr(launcher, "submit", lambda *_args: {"job_id": "123", "fixed_value_learning_rate": .001})
    monkeypatch.setattr(launcher, "ensure_report_timer", lambda *_args: "test.timer")
    result = launcher.advance(tmp_path, tmp_path / "python", "qwen")
    assert result == dict(stage="submitted", model="qwen", job_id="123",
                          fixed_value_learning_rate=.001, report_timer="test.timer")


def test_monitor_timer_is_idempotent(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="active\n")
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.ensure_report_timer(tmp_path, tmp_path / "root", "qwen") == (
        "query-cartridges-value-physical-qwen-report.timer")
    assert len(calls) == 1 and calls[0][:3] == ["systemctl", "--user", "is-active"]
