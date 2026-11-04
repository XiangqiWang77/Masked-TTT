#!/usr/bin/env python3
"""Prepare, but do not submit, one linked-selection + 32-token SD job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.babilong_linked_selection.long_rollout_protocol import prepare_development_manifest


CPU_TESTS = ["tests/test_linked_long_rollout_physical.py", "tests/test_linked_long_rollout_runtime.py",
             "tests/test_linked_long_rollout_protocol.py", "tests/test_linked_long_rollout_report.py",
             "tests/test_linked_long_rollout_prepare.py", "tests/test_linked_long_rollout_submit.py",
             "tests/test_linked_long_rollout_monitors.py", "tests/test_linked_long_rollout_holdout.py"]


def report_stub(workspace, root):
    references = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    baselines = [workspace / "query_cartridges_linked_selection_20261104",
                 workspace / "query_cartridges_linked_holdout_20261104"]
    arguments = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
                 "--baseline-root", str(baselines[0]), "--baseline-root", str(baselines[1]),
                 "--references", str(references), "--scorer-dir", str(root / "runtime/scorer"),
                 "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport sys\n"
            f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
            "from experiments.babilong_linked_selection.long_rollout_report import main\n"
            f"sys.argv = [sys.argv[0], *{arguments!r}]\nmain()\n")


def slurm_script(workspace, root):
    if any(ch.isspace() for ch in str(root)):
        raise ValueError("The Slurm log path cannot contain whitespace")
    runtime = root / "runtime"
    runtime_repo = runtime / "repo"
    driver = runtime / "rollout_driver.py"
    execution = runtime_repo / "experiments/babilong_linked_selection/long_rollout_execute.py"
    job_name = "qc-linked-roll32-hold" if "holdout" in root.name else "qc-linked-roll32"
    quote = lambda path: shlex.quote(str(path))
    return f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu,gpu_h100,gpu_h200,gpu_b200,gpu_rtx6000,scavenge_gpu
#SBATCH --constraint=gpu:a40|gpu:l40s|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --time-min=00:30:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --output={root}/logs/%j.out
#SBATCH --error={root}/logs/%j.err
set -euo pipefail
unset OPENAI_API_KEY OPENAI_API_KEY_FILE
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK_REPO={quote(runtime_repo)}
TASK_ROOT={quote(root)}
TASK_RUNTIME={quote(runtime)}
OPD_PYTHON=/home/da839/.conda/envs/TTT/bin/python
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eqi 'B200|Blackwell'; then
    OPD_PYTHON=/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/envs/ttt-cu128/bin/python
fi
export PYTHONPATH="$TASK_REPO/src:$TASK_REPO:$TASK_RUNTIME/helpers:$TASK_ROOT"
srun --kill-on-bad-exit=1 "$OPD_PYTHON" -c 'import torch; assert torch.cuda.device_count()==1'
srun --kill-on-bad-exit=1 "$OPD_PYTHON" -m pytest -q "$TASK_RUNTIME/gpu-tests/test_cuda_materialization.py"
srun --kill-on-bad-exit=1 "$OPD_PYTHON" -u {quote(execution)} --driver {quote(driver)} --job-root "$TASK_ROOT"
'''


def atomic_install(path, text):
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError("An existing job file differs: " + str(path))
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def refresh_validation(root, validation):
    """Allow a newer successful CPU test run before any submission."""
    if (root / "submission.json").exists():
        raise ValueError("Do not replace validation after submission")
    path = root / "cpu_validation_passed.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("submit_performed") is not False:
            raise ValueError("Existing validation is not an unsubmitted preparation")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(validation, indent=2) + "\n")
    temporary.replace(path)


def run_cpu_validation(python, repo=REPO):
    env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    check = subprocess.run([str(python), "-m", "pytest", "--noconftest", "-q", *CPU_TESTS], cwd=repo,
                           env=env, text=True, capture_output=True, timeout=180)
    if check.returncode:
        raise RuntimeError("Combined CPU validation failed:\n" + check.stdout + check.stderr)
    match = re.search(r"\b(\d+) passed\b", check.stdout)
    if not match:
        raise RuntimeError("The combined CPU validation did not report its passed test count")
    return {"cpu_tests_passed": True, "tests": CPU_TESTS, "passed_count": int(match.group(1)),
            "submit_performed": False, "online_api_calls": 0}


def validate_frozen_runtime(destination):
    required = (destination / "repo/src/masked_ttt", destination / "repo/experiments",
                destination / "repo/tests", destination / "rollout_driver.py",
                destination / "olmo_profile.json", destination / "helpers/time_budget.py",
                destination / "scorer/report_hybrid_partial.py",
                destination / "scorer/report_qwen_longbench_partial.py",
                destination / "gpu-tests/test_cuda_materialization.py")
    if any(not path.exists() for path in required):
        raise ValueError("The frozen runtime is incomplete")


def freeze_runtime(workspace, root):
    """Copy the small code/config runtime once; no model weights or scores."""
    destination = root / "runtime"
    if destination.exists():
        validate_frozen_runtime(destination)
        return destination
    staging = Path(tempfile.mkdtemp(prefix="runtime-staging-", dir=root))
    try:
        snapshot_repo = staging / "repo"
        snapshot_repo.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
        for name in ("src", "experiments", "tests"):
            shutil.copytree(REPO / name, snapshot_repo / name, ignore=ignore)
        for directory in ("helpers", "scorer", "gpu-tests"):
            (staging / directory).mkdir()
        files = {
            workspace / "query_cartridges_rollout32_olmo_dev_20261104/run.py": staging / "rollout_driver.py",
            workspace / "query_cartridges_value_lr_20261104/olmo_profile.json": staging / "olmo_profile.json",
            workspace / "gpu/runtime/time_budget.py": staging / "helpers/time_budget.py",
            workspace / "monitor/report_hybrid_partial.py": staging / "scorer/report_hybrid_partial.py",
            workspace / "monitor/report_qwen_longbench_partial.py": staging / "scorer/report_qwen_longbench_partial.py",
            workspace / "query_cartridges_value_lr_20261104/runtime/tests/test_cuda_materialization.py":
                staging / "gpu-tests/test_cuda_materialization.py",
        }
        for source, target in files.items():
            shutil.copy2(source, target)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_frozen_runtime(destination)
    return destination


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_linked_roll32_dev_20261104":
        raise ValueError("Use the dedicated development directory in the shared workspace")
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists():
            raise ValueError("Do not alter a submitted job")
        rollout = json.loads((workspace / "query_cartridges_rollout32_olmo_dev_20261104/manifest.json").read_text())
        linked = json.loads((workspace / "query_cartridges_linked_selection_20261104/manifest.json").read_text())
        profile = json.loads((workspace / "query_cartridges_value_lr_20261104/olmo_profile.json").read_text())
        manifest = prepare_development_manifest(rollout, linked, profile)
        runtime = freeze_runtime(workspace, root)
        (root / "logs").mkdir(exist_ok=True)
        atomic_install(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic_install(root / "report.py", report_stub(workspace, root))
        atomic_install(root / "job.sbatch", slurm_script(workspace, root))
        validation = run_cpu_validation(python, runtime / "repo")
        refresh_validation(root, validation)
        return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.workspace, args.job_root, args.python)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
