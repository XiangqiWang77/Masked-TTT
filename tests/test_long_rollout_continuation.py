from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import pytest

import experiments.long_rollout_steps.continuation as continuation
from experiments.long_rollout_steps.report import summarize
from test_long_rollout_holdout_report import fixture_holdout
from test_long_rollout_step_selection import fixture_development


def test_holdout_manifest_only_contains_development_eligible_models():
    development, report = fixture_development()
    for row in report["models"]["gpt"]["rows"]:
        row["teacher"] = row["step2"]
    report["models"]["gpt"]["overall"] = summarize(report["models"]["gpt"]["rows"])
    manifest, decisions = continuation.make_holdout_manifest(development, report)
    assert manifest["models"] == ["qwen"] and manifest["fixed_final_base_steps"] == {"qwen": 2}
    assert manifest["case_count"] == 800
    assert not set(manifest["case_ids"]) & set(manifest["development_case_ids"])
    assert not manifest["holdout_evaluated_at_selection"] and not decisions["gpt"]["eligible"]


def test_no_model_is_promoted_without_beating_the_teacher():
    development, report = fixture_development()
    for model in report["models"]:
        for row in report["models"][model]["rows"]:
            row["teacher"] = row["step2"]
        report["models"][model]["overall"] = summarize(report["models"][model]["rows"])
    manifest, choices = continuation.make_holdout_manifest(development, report)
    assert manifest is None and not any(item["eligible"] for item in choices.values())


def test_holdout_job_stays_one_gpu_and_uses_frozen_step_runner(tmp_path):
    script = continuation.holdout_slurm(tmp_path / "query_cartridges_longsteps_holdout_20261104")
    assert "#SBATCH --gpus-per-task=1" in script
    assert "holdout_execute.py" in script and "unset OPENAI_API_KEY" in script
    assert '--rootdir "$TASK_REPO" --confcutdir "$TASK_REPO"' in script


def test_holdout_is_prepared_with_its_selected_steps_and_reused(tmp_path, monkeypatch):
    development, report = fixture_development()
    manifest, _ = continuation.make_holdout_manifest(development, report)
    dev = tmp_path / "query_cartridges_longsteps_200dev_20261104"
    hold = tmp_path / "query_cartridges_longsteps_holdout_20261104"
    (dev / "runtime/repo").mkdir(parents=True)
    (dev / "finished.json").write_text('{}')
    (dev / "manifest.json").write_text(json.dumps(development))
    (dev / "latest.json").write_text(json.dumps(report))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="31 passed in 1s\n", stderr="")
    monkeypatch.setattr(continuation.subprocess, "run", run)
    continuation.prepare_holdout(tmp_path, dev, hold, tmp_path / "python", manifest)
    assert json.loads((hold / "manifest.json").read_text())["fixed_final_base_steps"] == {"qwen": 2, "gpt": 2}
    assert json.loads((hold / "cpu_validation_passed.json").read_text())["passed_count"] == 31
    assert (hold / "runtime/repo/experiments/long_rollout_steps/holdout_runner.py").is_file()
    assert "holdout_execute.py" in (hold / "job.sbatch").read_text()
    assert "--rootdir" in calls[0] and "--confcutdir" in calls[0]
    (hold / "submission.json").write_text('{"job_id": "123"}')
    continuation.prepare_holdout(tmp_path, dev, hold, tmp_path / "python", manifest)
    assert len(calls) == 1


def test_holdout_submission_honors_the_gpu_cap_and_is_idempotent(tmp_path, monkeypatch):
    development, report = fixture_development()
    manifest, _ = continuation.make_holdout_manifest(development, report)
    root = tmp_path / "query_cartridges_longsteps_holdout_20261104"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "cpu_validation_passed.json").write_text(json.dumps(
        {"passed": True, "steps": manifest["fixed_final_base_steps"]}))
    (root / "job.sbatch").write_text("test")
    @contextmanager
    def slot():
        yield True
    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=slot))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345\n")
    monkeypatch.setattr(continuation.subprocess, "run", run)
    result = continuation.submit_holdout(tmp_path, root)
    assert result["gpus"] == 1 and result["cases_per_model"] == 800
    assert result["fixed_final_base_steps"] == {"qwen": 2, "gpt": 2}
    assert continuation.submit_holdout(tmp_path, root) == result and len(calls) == 1


def test_advance_does_not_submit_before_pilot_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation.extend, "prepare", lambda *_args, **_kwargs: pytest.fail("early submit"))
    assert continuation.advance(tmp_path, tmp_path / "python")["stage"] == "pilot_in_progress"


def test_advance_waits_for_200_dev_before_any_holdout_submission(tmp_path, monkeypatch):
    pilot = tmp_path / "query_cartridges_longsteps_pilot_20261104"
    dev = tmp_path / "query_cartridges_longsteps_200dev_20261104"
    pilot.mkdir(); dev.mkdir()
    (pilot / "finished.json").write_text('{}')
    monkeypatch.setattr(continuation, "_pilot_snapshot", lambda *_args: {})
    monkeypatch.setattr(continuation.extend, "prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(continuation.extend, "submit", lambda *_args: {"job_id": "101"})
    monkeypatch.setattr(continuation, "_refresh", lambda _root: {})
    monkeypatch.setattr(continuation, "prepare_holdout", lambda *_args: pytest.fail("early holdout"))
    assert continuation.advance(tmp_path, tmp_path / "python") == {
        "stage": "development_in_progress", "job_id": "101"}


def test_advance_combines_complete_holdout_with_the_preselected_steps(tmp_path, monkeypatch):
    dev_manifest, development, _fixture_manifest, holdout = fixture_holdout(tmp_path)
    pilot = tmp_path / "query_cartridges_longsteps_pilot_20261104"
    dev = tmp_path / "query_cartridges_longsteps_200dev_20261104"
    hold = tmp_path / "query_cartridges_longsteps_holdout_20261104"
    for root in (pilot, dev, hold):
        root.mkdir(); (root / "finished.json").write_text('{}')
    (dev / "manifest.json").write_text(json.dumps(dev_manifest))
    monkeypatch.setattr(continuation, "_pilot_snapshot", lambda *_args: {})
    monkeypatch.setattr(continuation.extend, "prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(continuation.extend, "submit", lambda *_args: {"job_id": "101"})
    monkeypatch.setattr(continuation, "prepare_holdout", lambda *_args: hold)
    monkeypatch.setattr(continuation, "submit_holdout", lambda *_args: {"job_id": "102"})
    monkeypatch.setattr(continuation, "_refresh", lambda root: development if root == dev else holdout)
    result = continuation.advance(tmp_path, tmp_path / "python")
    assert result["stage"] == "complete" and set(result["models"]) == {"qwen", "gpt"}
    combined = json.loads((dev / "combined_1000.json").read_text())
    assert combined["fixed_final_base_steps"] == {"qwen": 2, "gpt": 2}
    assert combined["models"]["qwen"]["overall"]["n"] == 1000
