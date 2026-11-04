#!/usr/bin/env python3
"""Freeze and validate the one-GPU query-head gain development experiment."""
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
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition

CPU_TESTS = ["tests/test_query_head_prepare.py", "tests/test_query_head_monitor.py", "tests/test_query_head_runner.py", "tests/test_query_head_integration.py",
             "tests/test_gqa_head_gain.py", "tests/test_batched_sd.py", "tests/test_memory.py",
             "tests/test_materialization_equivalence.py", "tests/test_kv_only_cartridge.py",
             "tests/test_gpt_request_date.py"]


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def make_driver(source):
    """Use the established runner with the original eight-token rollout cap."""
    replacements = {
        "manifest.get('student_rollout_tokens')!=32": "manifest.get('student_rollout_tokens')!=8",
        "t['student_rollout_limit']!=32": "t['student_rollout_limit']!=8",
        "student_rollout_tokens=32,": "student_rollout_tokens=8,",
        "Original SD / 32-token rollout / independent cartridge contract failed":
            "Original SD / 8-token rollout / independent cartridge contract failed",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("The frozen rollout driver no longer has the expected setting: " + old)
        source = source.replace(old, new, 1)
    if 'MASKED_TTT_FIXED_CHAT_DATE' not in source:
        raise ValueError("The benchmark request date must be fixed across resumed cases")
    return source


def report_stub(workspace, root, baseline):
    refs = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    arguments = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
                 "--baseline-root", str(baseline), "--references", str(refs),
                 "--scorer-dir", str(root / "runtime/scorer"), "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport sys\n"
            f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
            "from experiments.gqa_head_gain.report import main\n"
            f"sys.argv = [sys.argv[0], *{arguments!r}]\nmain()\n")


def slurm_script(root):
    runtime = root / "runtime"
    repo = runtime / "repo"
    job_name = "qc-gqa-gain-hold" if "holdout" in root.name else "qc-gqa-gain-dev"
    q = lambda path: shlex.quote(str(path))
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
TASK_REPO={q(repo)}
TASK_ROOT={q(root)}
TASK_RUNTIME={q(runtime)}
OPD_PYTHON=/home/da839/.conda/envs/TTT/bin/python
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eqi 'B200|Blackwell'; then
    OPD_PYTHON=/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/envs/ttt-cu128/bin/python
fi
export PYTHONPATH="$TASK_REPO/src:$TASK_REPO:$TASK_RUNTIME/helpers:$TASK_ROOT"
srun --kill-on-bad-exit=1 "$OPD_PYTHON" -c 'import torch; assert torch.cuda.device_count()==1'
srun --kill-on-bad-exit=1 "$OPD_PYTHON" -m pytest -q "$TASK_REPO/tests/test_cuda_materialization.py"
srun --kill-on-bad-exit=1 "$OPD_PYTHON" -u "$TASK_REPO/experiments/gqa_head_gain/execute.py" --job-root "$TASK_ROOT"
'''


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_gqa_gain_dev_20261104":
        raise ValueError("Use the dedicated development directory")
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("Do not change a submitted or evaluated experiment")
        linked = json.loads((workspace / "query_cartridges_linked_selection_20261104/manifest.json").read_text())
        rollout_root = workspace / "query_cartridges_rollout32_olmo_dev_20261104"
        rollout = json.loads((rollout_root / "manifest.json").read_text())
        profile_root = workspace / "query_cartridges_value_lr_20261104"
        profile = json.loads((profile_root / "olmo_profile.json").read_text())
        development, holdout = linked["case_ids"], linked["holdout_case_ids"]
        validate_partition(development, holdout)
        if (development != rollout["case_ids"] or holdout != rollout["holdout_case_ids"] or
            linked["model_protocols"] != rollout["model_protocols"] or linked["keep_fraction"] != .1 or
            linked["selection_policy"] != "decision_entity_link_v1" or linked["entity_reserve_fraction"] != .25 or
            linked["selected_transfer_weight"] != 0 or profile["student_rollout_tokens"] != 8 or
            profile["steps"] != 2 or profile["gain_learning_rate"] != .01 or
            profile["value_learning_rate"] != .0001 or profile["self_question_count"] != 2):
            raise ValueError("The reference settings differ from the original OLMo study")
        manifest = dict(rollout, mode="development", case_count=200, case_ids=list(development),
            development_case_ids=list(development), holdout_case_ids=list(holdout), models=["qwen", "gpt"],
            scope="OLMo-style linked evidence and per-query-head gains", gain_head_mode="query",
            selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25,
            student_rollout_tokens=8, max_gpu_hours=36, online_api_calls=0,
            development_gate="Beat same-run full, selected-zero, and Teacher on all 200 cases",
            holdout_gate="At least +3 points vs same-run Teacher and positive vs full and selected-zero")
        runtime = root / "runtime"
        if runtime.exists():
            shutil.rmtree(runtime)
        runtime.mkdir()
        frozen_repo = runtime / "repo"
        frozen_repo.mkdir()
        ignored = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
        for folder in ("src", "experiments", "tests"):
            shutil.copytree(REPO / folder, frozen_repo / folder, ignore=ignored)
        atomic(runtime / "rollout_driver.py", make_driver((rollout_root / "run.py").read_text()))
        shutil.copy2(profile_root / "olmo_profile.json", runtime / "olmo_profile.json")
        shared_runtime = workspace / "query_cartridges_linked_roll32_dev_20261104/runtime"
        shutil.copytree(shared_runtime / "helpers", runtime / "helpers", ignore=ignored)
        shutil.copytree(shared_runtime / "scorer", runtime / "scorer", ignore=ignored)
        (root / "logs").mkdir(exist_ok=True)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "report.py", report_stub(workspace, root,
                                                workspace / "query_cartridges_linked_selection_20261104"))
        atomic(root / "job.sbatch", slurm_script(root))
        env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        check = subprocess.run([str(python), "-m", "pytest", "-q", *CPU_TESTS], cwd=frozen_repo,
                               env=env, text=True, capture_output=True, timeout=180)
        if check.returncode:
            raise RuntimeError("The frozen CPU validation failed:\n" + check.stdout + check.stderr)
        match = re.search(r"\b(\d+) passed\b", check.stdout)
        if not match:
            raise RuntimeError("The CPU validation did not report its passed test count")
        result = dict(cpu_tests_passed=True, passed_count=int(match.group(1)), tests=CPU_TESTS,
                      gpu_preflight_required=True, submitted=False, online_api_calls=0)
        atomic(root / "cpu_validation_passed.json", json.dumps(result, indent=2) + "\n")
        return result


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace:
        raise ValueError("The job must remain in the shared workspace")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        validation = json.loads((root / "cpu_validation_passed.json").read_text())
        if not validation.get("cpu_tests_passed"):
            raise ValueError("CPU validation has not passed")
        manifest = json.loads((root / "manifest.json").read_text())
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU cap is full")
            result = subprocess.run(["sbatch", "--parsable", "--comment=olmo-query-head-comparison",
                                     str(root / "job.sbatch")], check=True, text=True,
                                    capture_output=True, timeout=30)
            job = result.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, mode=manifest["mode"], models=manifest["models"],
                        cases_per_model=manifest["case_count"], cached_scores=True, online_api_calls=0,
                        gain_head_mode="query", student_rollout_tokens=8, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.job_root, args.python), indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace, args.job_root), indent=2))
        from experiments.gqa_head_gain.monitor import enable_monitor
        enable_monitor(args.workspace.resolve(), args.job_root.resolve())
        print("Hourly reporting and source-disjoint promotion enabled")


if __name__ == "__main__":
    main()
