#!/usr/bin/env python3
"""Promote the frozen combined study to 800 source-disjoint cases if eligible."""
import argparse
import fcntl
import getpass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.babilong_linked_selection.long_rollout_protocol import prepare_holdout_manifest
from experiments.babilong_linked_selection.prepare_long_rollout_job import (
    atomic_install, refresh_validation, report_stub, run_cpu_validation, slurm_script,
    validate_frozen_runtime,
)
from experiments.babilong_linked_selection.submit_long_rollout_job import atomic


def freeze(workspace, development_root):
    """Freeze only a complete development report, without scheduling GPUs."""
    workspace, development_root = workspace.resolve(), development_root.resolve()
    if (development_root.parent != workspace or
        development_root.name != "query_cartridges_linked_roll32_dev_20261104"):
        raise ValueError("Unexpected combined development directory")
    if not (development_root / "finished.json").is_file():
        return None
    development = json.loads((development_root / "manifest.json").read_text())
    root = workspace / "query_cartridges_linked_roll32_holdout_20261104"
    root.mkdir(exist_ok=True)
    with (root / "promotion.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        snapshot, manifest_path = root / "development_snapshot.json", root / "manifest.json"
        report = json.loads((snapshot if snapshot.is_file() else development_root / "latest.json").read_text())
        try:
            manifest = prepare_holdout_manifest(development, report)
        except RuntimeError:
            return None
        atomic_install(snapshot, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if not manifest["models"]:
            marker = root / "no_eligible_model.json"
            atomic_install(marker, json.dumps({"reason": "no_model_beats_all_development_controls",
                "decisions": manifest["development_decisions"]}, indent=2) + "\n")
            return {"state": "not_eligible", "root": root, "manifest": manifest}
        source_runtime, holdout_runtime = development_root / "runtime", root / "runtime"
        validate_frozen_runtime(source_runtime)
        if not holdout_runtime.exists():
            shutil.copytree(source_runtime, holdout_runtime,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        validate_frozen_runtime(holdout_runtime)
        (root / "logs").mkdir(exist_ok=True)
        atomic_install(manifest_path, json.dumps(manifest, indent=2) + "\n")
        atomic_install(root / "report.py", report_stub(workspace, root))
        atomic_install(root / "job.sbatch", slurm_script(workspace, root))
        return {"state": "prepared", "root": root, "manifest": manifest}


def submit(workspace, development_root):
    """Run validation and submit exactly one single-GPU holdout if qualified."""
    prepared = freeze(workspace, development_root)
    if prepared is None:
        return {"state": "waiting"}
    if prepared["state"] != "prepared":
        return {"state": prepared["state"]}
    workspace, root = workspace.resolve(), prepared["root"]
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.is_file():
            job = json.loads(target.read_text())["job_id"]
        else:
            validation = run_cpu_validation(Path("/home/da839/.conda/envs/TTT/bin/python"), root / "runtime/repo")
            refresh_validation(root, validation)
            with submission_slot() as available:
                if not available:
                    return {"state": "waiting_slot"}
                current = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j"],
                                         check=True, capture_output=True, text=True, timeout=20)
                jobs = [parts[0].strip() for line in current.stdout.splitlines()
                        if len(parts := line.split("|", 1)) == 2 and
                        parts[1].strip() == "qc-linked-roll32-hold"]
                if len(jobs) > 1:
                    raise RuntimeError("More than one combined holdout job exists")
                if jobs:
                    job = jobs[0]
                else:
                    sent = subprocess.run(["sbatch", "--parsable", "--comment=cartridge-linked-roll32-holdout",
                                           str(root / "job.sbatch")], check=True, capture_output=True,
                                          text=True, timeout=30)
                    job = sent.stdout.strip().split(";", 1)[0]
                if not job.isdigit():
                    raise ValueError("Invalid scheduler response")
                atomic(target, {"job_id": job, "gpus": 1, "models": prepared["manifest"]["models"],
                    "cases_per_model": 800, "keep_fraction": .1, "student_rollout_tokens": 32,
                    "selected_transfer_weight": 0, "online_api_calls": 0, "max_gpu_hours": 36,
                    "max_hours_per_allocation": 6, "created": time.time()})
    from experiments.babilong_linked_selection.long_rollout_monitors import enable_timer
    enable_timer(workspace, root, "holdout")
    return {"state": "submitted", "job_id": job}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--development-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(submit(args.workspace, args.development_root)))


if __name__ == "__main__":
    main()
