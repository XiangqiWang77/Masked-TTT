from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import experiments.value_lr_physical.prepare as implementation
from experiments.value_lr_physical.prepare import physical_driver, physical_slurm, prepare, submit
from test_value_lr_physical import fixture_value_development


def driver_source():
    return "\n".join(["manifest.get('student_rollout_tokens')!=32", "t['student_rollout_limit']!=32",
        "student_rollout_tokens=32,", "Original SD / 32-token rollout / independent cartridge contract failed",
        "MASKED_TTT_FIXED_CHAT_DATE", "precision=CROSS/'query_cartridges_precision_probe_20261104'"])


def test_frozen_driver_and_job_keep_the_original_method_and_one_gpu(tmp_path):
    source = physical_driver(driver_source())
    assert "student_rollout_tokens=8," in source and "MASKED_TTT_FIXED_CHAT_DATE" in source
    assert "precision=ROOT/'disabled_optional_probe'" in source
    script = physical_slurm(tmp_path / "query_cartridges_value_lr_physical_gpt_holdout_20261104", "gpt")
    assert "#SBATCH --gpus-per-task=1" in script and "value_lr_physical/execute.py" in script
    assert "unset OPENAI_API_KEY" in script and '--rootdir "$TASK_REPO"' in script
    assert "--constraint=gpu:a40" not in script
    assert "--partition=gpu_rtx6000,gpu_b200,gpu_h200,gpu_h100,scavenge_gpu" in script
    with pytest.raises(ValueError):
        physical_driver(driver_source().replace("MASKED_TTT_FIXED_CHAT_DATE", ""))


def test_prepare_freezes_only_the_completed_models_selected_configuration(tmp_path, monkeypatch):
    development, report, profile = fixture_value_development()
    dev = tmp_path / "query_cartridges_value_lr_20261104"
    dev.mkdir()
    for name, data in (("manifest.json", development), ("latest.json", report), ("olmo_profile.json", profile)):
        (dev / name).write_text(json.dumps(data))
    old = tmp_path / "query_cartridges_rollout32_olmo_dev_20261104"
    old.mkdir()
    (old / "run.py").write_text(driver_source())
    shared = tmp_path / "query_cartridges_linked_roll32_dev_20261104/runtime"
    (shared / "helpers").mkdir(parents=True)
    (shared / "scorer").mkdir()
    root = tmp_path / "query_cartridges_value_lr_physical_gpt_holdout_20261104"
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="40 passed in 1s\n", stderr="")
    monkeypatch.setattr(implementation.subprocess, "run", run)
    checks = prepare(tmp_path, root, tmp_path / "python", "gpt")
    assert checks["cpu_tests_passed"] and checks["fixed_value_learning_rate"] == .001
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["models"] == ["gpt"] and manifest["case_count"] == 800
    assert (root / "development_snapshot.json").is_file()
    assert '--rootdir' in calls[0] and (root / "runtime/rollout_driver.py").is_file()
    assert prepare(tmp_path, root, tmp_path / "python", "gpt") == checks
    assert len(calls) == 1


def test_failed_cpu_check_retries_same_frozen_runtime_without_resubmitting(tmp_path, monkeypatch):
    development, report, profile = fixture_value_development()
    dev = tmp_path / "query_cartridges_value_lr_20261104"
    dev.mkdir()
    for name, data in (("manifest.json", development), ("latest.json", report), ("olmo_profile.json", profile)):
        (dev / name).write_text(json.dumps(data))
    old = tmp_path / "query_cartridges_rollout32_olmo_dev_20261104"
    old.mkdir()
    (old / "run.py").write_text(driver_source())
    shared = tmp_path / "query_cartridges_linked_roll32_dev_20261104/runtime"
    (shared / "helpers").mkdir(parents=True)
    (shared / "scorer").mkdir()
    root = tmp_path / "query_cartridges_value_lr_physical_gpt_holdout_20261104"
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="temporary resource failure")
        return SimpleNamespace(returncode=0, stdout="40 passed in 1s\n", stderr="")
    monkeypatch.setattr(implementation.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="temporary resource failure"):
        prepare(tmp_path, root, tmp_path / "python", "gpt")
    marker = root / "runtime/repo/frozen_marker"
    marker.write_text("preserved")
    manifest_path = root / "manifest.json"
    manifest_text = manifest_path.read_text()
    altered = json.loads(manifest_text)
    altered["fixed_value_learning_rate"] = .0003
    manifest_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="different protocol"):
        prepare(tmp_path, root, tmp_path / "python", "gpt")
    assert len(calls) == 1
    manifest_path.write_text(manifest_text)
    checks = prepare(tmp_path, root, tmp_path / "python", "gpt")
    assert checks["cpu_tests_passed"] and checks["fixed_value_learning_rate"] == .001
    assert marker.read_text() == "preserved" and len(calls) == 2
    assert not (root / "submission.json").exists()


def test_submission_is_single_gpu_guarded_and_idempotent(tmp_path, monkeypatch):
    development, report, profile = fixture_value_development()
    from experiments.value_lr_physical.protocol import holdout_manifest
    manifest = holdout_manifest(development, report, profile, "gpt")
    root = tmp_path / "query_cartridges_value_lr_physical_gpt_holdout_20261104"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "cpu_validation_passed.json").write_text(json.dumps(
        {"cpu_tests_passed": True, "model": "gpt", "fixed_value_learning_rate": .001}))
    (root / "job.sbatch").write_text("test")
    @contextmanager
    def slot():
        yield True
    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=slot))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345\n")
    monkeypatch.setattr(implementation.subprocess, "run", run)
    result = submit(tmp_path, root)
    assert result["job_id"] == "12345" and result["gpus"] == 1 and result["cases"] == 800
    assert submit(tmp_path, root) == result and len(calls) == 1
