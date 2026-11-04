from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import pytest

import experiments.step_physical.prepare as implementation
from experiments.step_physical.prepare import job_script, prepare, submit
from test_step_physical import fixture_development
from test_value_lr_physical_prepare import driver_source


def setup_workspace(tmp_path):
    development, report, profile = fixture_development()
    dev = tmp_path / "query_cartridges_step_dev_20261104"
    dev.mkdir()
    (dev / "finished.json").write_text("{}")
    (dev / "manifest.json").write_text(json.dumps(development))
    (dev / "latest.json").write_text(json.dumps(report))
    profile_root = tmp_path / "query_cartridges_value_lr_20261104"
    profile_root.mkdir()
    (profile_root / "olmo_profile.json").write_text(json.dumps(profile))
    driver = tmp_path / "query_cartridges_rollout32_olmo_dev_20261104"
    driver.mkdir()
    (driver / "run.py").write_text(driver_source())
    shared = tmp_path / "query_cartridges_linked_roll32_dev_20261104/runtime"
    (shared / "helpers").mkdir(parents=True)
    (shared / "scorer").mkdir()
    return tmp_path / "query_cartridges_step_physical_gpt_holdout_20261104"


def test_job_is_single_gpu_high_memory_and_uses_physical_step_driver(tmp_path):
    root = tmp_path / "query_cartridges_step_physical_gpt_holdout_20261104"
    script = job_script(root)
    assert "#SBATCH --gpus-per-task=1" in script and "step_physical/execute.py" in script
    assert "--constraint=gpu:a40" not in script and "unset OPENAI_API_KEY" in script
    assert '--rootdir "$TASK_REPO"' in script


def test_preparation_freezes_the_development_choice_and_retries_validation(tmp_path, monkeypatch):
    root = setup_workspace(tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="temporary failure")
        return SimpleNamespace(returncode=0, stdout="50 passed in 1s\n", stderr="")
    monkeypatch.setattr(implementation.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="temporary failure"):
        prepare(tmp_path, root, tmp_path / "python")
    marker = root / "runtime/repo/frozen_marker"
    marker.write_text("preserved")
    checks = prepare(tmp_path, root, tmp_path / "python")
    assert checks["cpu_tests_passed"] and checks["fixed_base_steps"] == 8
    assert calls[-1][1]["env"]["CARTRIDGE_TEST_BASE_STEPS"] == "8"
    assert marker.read_text() == "preserved" and len(calls) == 2
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["models"] == ["gpt"] and manifest["case_count"] == 800
    assert manifest["steps"] == 8 and manifest["fixed_value_learning_rate"] == .0001
    assert prepare(tmp_path, root, tmp_path / "python") == checks and len(calls) == 2


def test_submission_is_idempotent_and_obeys_the_shared_gpu_limit(tmp_path, monkeypatch):
    root = setup_workspace(tmp_path)
    monkeypatch.setattr(implementation.subprocess, "run", lambda *_args, **_kwargs:
        SimpleNamespace(returncode=0, stdout="50 passed in 1s\n", stderr=""))
    prepare(tmp_path, root, tmp_path / "python")
    @contextmanager
    def slot():
        yield True
    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=slot))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345\n")
    monkeypatch.setattr(implementation.subprocess, "run", run)
    first = submit(tmp_path, root)
    assert first["job_id"] == "12345" and first["gpus"] == 1 and first["fixed_base_steps"] == 8
    assert submit(tmp_path, root) == first and len(calls) == 1
