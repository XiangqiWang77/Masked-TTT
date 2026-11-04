#!/usr/bin/env python3
"""Freeze and submit one low-priority GPU-only transfer diagnostic."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.gqa_head_gain.prepare import atomic, slurm_script
from experiments.long_rollout_steps.extend import isolate_gpu_preflight

CPU_TESTS = ["tests/test_gain_transfer_probe.py", "tests/test_gain_transfer_job.py",
             "tests/test_gain_transfer_prepare.py", "tests/test_gain_transfer_report.py", "tests/test_gain_transfer_monitor.py",
             "tests/test_freeze_readout_summary.py", "tests/test_kv_only_cartridge.py",
             "tests/test_gpt_request_date.py"]


def make_manifest(workspace):
    development = workspace / "query_cartridges_value_lr_20261104"
    source = json.loads((development / "manifest.json").read_text())
    anchor = json.loads((workspace / "query_cartridges_basis_dev_20261104/manifest.json").read_text())
    profile = json.loads((development / "olmo_profile.json").read_text())
    keys = anchor["case_ids"]
    sources = {}
    for key in keys:
        parts = key.split("-", 2)
        if len(parts) != 3 or parts[0] != "babilong" or parts[1] not in ("8", "16", "32", "64"):
            raise ValueError("Unexpected development case")
        sources.setdefault(parts[2], set()).add(parts[1])
    if (len(keys) != 40 or len(set(keys)) != 40 or len(sources) != 10 or
        any(lengths != {"8", "16", "32", "64"} for lengths in sources.values()) or
        not set(keys).issubset(source["case_ids"]) or source["keep_fraction"] != .1 or
        source["reference_answers_used"] is not False or source["selected_transfer_weight"] != 0 or
        profile["steps"] != 2 or profile["student_rollout_tokens"] != 8 or
        profile["value_learning_rate"] != .0001 or profile["gain_learning_rate"] != .01 or
        set(source["model_protocols"]) != {"qwen", "gpt"}):
        raise ValueError("The frozen OLMo-style diagnostic inputs changed")
    manifest = dict(scope="40 development cases per backbone; read-only gain and first-token diagnostics",
        case_ids=keys, models=["qwen", "gpt"], model_protocols=source["model_protocols"],
        keep_fraction=.1, selected_transfer_weight=0., reference_answers_used=False,
        online_api_calls=0, random_order_seed=20261104, max_gpu_hours=6,
        diagnosis_only=True, same_olmo_objective=True)
    return manifest, profile


def make_slurm(root):
    script = slurm_script(root)
    changes = {
        "qc-gqa-gain-dev": "qc-gain-transfer",
        "gqa_head_gain/execute.py": "value_lr_physical/run_gain_transfer.py",
        "#SBATCH --partition=gpu,gpu_h100,gpu_h200,gpu_b200,gpu_rtx6000,scavenge_gpu":
            "#SBATCH --partition=gpu_h200,gpu_h100,gpu_b200,gpu_rtx6000,scavenge_gpu",
        "#SBATCH --constraint=gpu:a40|gpu:l40s|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell":
            "#SBATCH --constraint=gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell\n#SBATCH --nice=10000",
    }
    for old, new in changes.items():
        if script.count(old) != 1:
            raise ValueError("Unexpected single-GPU script")
        script = script.replace(old, new, 1)
    return isolate_gpu_preflight(script)


def validate_frozen(root, python, manifest, profile):
    runtime, frozen = root / "runtime", root / "runtime/repo"
    if (not (root / "manifest.json").exists() or json.loads((root / "manifest.json").read_text()) != manifest or
        not (root / "olmo_profile.json").exists() or
        json.loads((root / "olmo_profile.json").read_text()) != profile or not (root / "job.sbatch").exists() or
        not (runtime / "helpers/time_budget.py").exists() or
        any(not (frozen / name).exists() for name in ("src", "experiments", "tests"))):
        raise ValueError("The diagnostic runtime is incomplete or uses another protocol")
    env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    checked = subprocess.run([str(python), "-m", "pytest", "--rootdir", str(frozen),
        "--confcutdir", str(frozen), "-q", *CPU_TESTS], cwd=frozen, env=env,
        text=True, capture_output=True, timeout=180)
    match = re.search(r"\b(\d+) passed\b", checked.stdout)
    if checked.returncode or not match:
        raise RuntimeError("Frozen diagnostic tests failed:\n" + checked.stdout + checked.stderr)
    result = dict(cpu_tests_passed=True, passed_count=int(match.group(1)), tests=CPU_TESTS,
                  loaded_model_gpu_preflight_required=True, online_api_calls=0, created=time.time())
    atomic(root / "cpu_validation_passed.json", json.dumps(result, indent=2) + "\n")
    return result


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_gain_transfer_20261104":
        raise ValueError("Use the dedicated transfer-diagnostic directory")
    manifest, profile = make_manifest(workspace)
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("Do not modify a submitted diagnostic")
        runtime = root / "runtime"
        if runtime.exists():
            return validate_frozen(root, python, manifest, profile)
        runtime.mkdir()
        frozen = runtime / "repo"
        frozen.mkdir()
        ignored = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
        for folder in ("src", "experiments", "tests"):
            shutil.copytree(REPO / folder, frozen / folder, ignore=ignored)
        helpers = workspace / "query_cartridges_linked_roll32_dev_20261104/runtime/helpers"
        shutil.copytree(helpers, runtime / "helpers", ignore=ignored)
        (root / "logs").mkdir(exist_ok=True)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "olmo_profile.json", json.dumps(profile, indent=2) + "\n")
        atomic(root / "job.sbatch", make_slurm(root))
        return validate_frozen(root, python, manifest, profile)


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_gain_transfer_20261104":
        raise ValueError("The diagnostic must use its dedicated directory")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        validation = json.loads((root / "cpu_validation_passed.json").read_text())
        if not validation.get("cpu_tests_passed"):
            raise ValueError("Frozen diagnostic tests have not passed")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU cap is full")
            result = subprocess.run(["sbatch", "--parsable", "--comment=original-sd-transfer-diagnostic",
                                     str(root / "job.sbatch")], check=True, capture_output=True, text=True, timeout=30)
            job = result.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, models=["qwen", "gpt"], cases_per_model=40,
                        cached_scores=True, online_api_calls=0, low_priority=True, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "job-root", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.job_root, args.python), indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace, args.job_root), indent=2))


if __name__ == "__main__":
    main()
