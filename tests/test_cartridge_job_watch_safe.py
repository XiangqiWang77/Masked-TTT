from types import SimpleNamespace
import json
import subprocess

import pytest

from experiments.value_lr_physical import cartridge_job_watch as watch


def response(stdout="", *, code=0, stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


def queue_fixture(monkeypatch, results):
    iterator = iter(results)
    seen = []
    def fake(command, **kwargs):
        seen.append(command[0])
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(watch.subprocess, "run", fake)
    return seen


def test_live_or_unobservable_job_is_not_terminal(monkeypatch):
    seen = queue_fixture(monkeypatch, [response("123\n")])
    assert watch.observed_terminal("123") is None and seen == ["squeue"]
    seen = queue_fixture(monkeypatch, [response(code=1)])
    assert watch.observed_terminal("123") is None and seen == ["squeue"]
    queue_fixture(monkeypatch, [subprocess.TimeoutExpired("squeue", 20)])
    assert watch.observed_terminal("123") is None


@pytest.mark.parametrize("control", ["JobId=123 JobState=PENDING Requeue=1",
                                     "JobId=123 JobState=PREEMPTED Requeue=1",
                                     "JobId=123 JobState=NODE_FAIL Requeue=1",
                                     "JobId=123 JobState=FAILED Requeue=1",
                                     "JobId=123 JobState=OUT_OF_MEMORY Requeue=1",
                                     "unparseable response"])
def test_requeue_or_unknown_control_state_is_not_terminal(monkeypatch, control):
    queue_fixture(monkeypatch, [response(), response(control)])
    assert watch.observed_terminal("123") is None


def test_scheduler_terminal_state_can_be_recorded(monkeypatch, tmp_path):
    queue_fixture(monkeypatch, [response(), response("JobId=123 JobState=OUT_OF_MEMORY Requeue=0")])
    (tmp_path / "submission.json").write_text(json.dumps(dict(job_id="123")))
    assert watch.record_terminal(tmp_path) == "OUT_OF_MEMORY"
    assert json.loads((tmp_path / "terminal_job_state.json").read_text())["job_id"] == "123"


def test_missing_handle_needs_accounting_and_a_second_queue_check(monkeypatch):
    invalid = response(code=1, stderr="slurm_load_jobs error: Invalid job id specified")
    history = response("123.batch|COMPLETED\n123|FAILED\n")
    queue_fixture(monkeypatch, [response(), invalid, history, response()])
    assert watch.observed_terminal("123") == "FAILED"
    queue_fixture(monkeypatch, [response(), invalid, history, response("123\n")])
    assert watch.observed_terminal("123") is None
    queue_fixture(monkeypatch, [response(), invalid, response("123|FAILED\n123|RUNNING\n")])
    assert watch.observed_terminal("123") is None


def test_control_error_or_missing_accounting_is_not_terminal(monkeypatch):
    queue_fixture(monkeypatch, [response(), response(code=1, stderr="temporary scheduler error")])
    assert watch.observed_terminal("123") is None
    invalid = response(code=1, stderr="Invalid job id specified")
    queue_fixture(monkeypatch, [response(), invalid, response()])
    assert watch.observed_terminal("123") is None
