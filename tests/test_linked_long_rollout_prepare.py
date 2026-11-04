import json
from pathlib import Path

import pytest

from experiments.babilong_linked_selection.prepare_long_rollout_job import (
    atomic_install, freeze_runtime, refresh_validation, report_stub, slurm_script,
)


def test_generated_job_uses_one_gpu_and_no_online_decision():
    workspace = Path("/scratch/stack/cross")
    root = workspace / "query_cartridges_linked_roll32_dev_20261104"
    script = slurm_script(workspace, root)
    assert "#SBATCH --gpus-per-task=1" in script
    assert "unset OPENAI_API_KEY OPENAI_API_KEY_FILE" in script
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in script
    assert "long_rollout_execute.py" in script and "--job-root" in script
    assert "sbatch " not in script
    report = report_stub(workspace, root)
    compile(report, "report.py", "exec")
    assert "query_cartridges_linked_selection_20261104" in report
    assert "query_cartridges_linked_holdout_20261104" in report


def test_preparation_does_not_replace_different_existing_job_files(tmp_path):
    path = tmp_path / "job.sbatch"
    atomic_install(path, "same\n")
    atomic_install(path, "same\n")
    with pytest.raises(FileExistsError): atomic_install(path, "different\n")
    assert path.read_text() == "same\n"


def test_validation_refresh_only_before_submission(tmp_path):
    refresh_validation(tmp_path, {"cpu_tests_passed": True, "submit_performed": False, "passed_count": 3})
    refresh_validation(tmp_path, {"cpu_tests_passed": True, "submit_performed": False, "passed_count": 4})
    path = tmp_path / "cpu_validation_passed.json"
    assert json.loads(path.read_text())["passed_count"] == 4
    (tmp_path / "submission.json").write_text("{}")
    with pytest.raises(ValueError): refresh_validation(tmp_path, {})


def test_runtime_snapshot_is_stable_and_contains_no_model_weights(tmp_path):
    workspace, root = tmp_path / "workspace", tmp_path / "job"
    root.mkdir()
    inputs = ("query_cartridges_rollout32_olmo_dev_20261104/run.py",
              "query_cartridges_value_lr_20261104/olmo_profile.json", "gpu/runtime/time_budget.py",
              "monitor/report_hybrid_partial.py", "monitor/report_qwen_longbench_partial.py",
              "query_cartridges_value_lr_20261104/runtime/tests/test_cuda_materialization.py")
    for name in inputs:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    frozen = freeze_runtime(workspace, root)
    assert (frozen / "rollout_driver.py").read_text() == "original\n"
    (workspace / inputs[0]).write_text("changed later\n")
    assert freeze_runtime(workspace, root) == frozen
    assert (frozen / "rollout_driver.py").read_text() == "original\n"
    assert not list(frozen.rglob("*.safetensors"))
