#!/usr/bin/env python3
"""Submit one prepared combined job after its two component studies finish."""
import argparse
import fcntl
import getpass
import json
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.babilong_linked_selection.long_rollout_protocol import prepare_development_manifest
from experiments.babilong_linked_selection.prepare_long_rollout_job import CPU_TESTS, report_stub, slurm_script


def finished_component(status):
    return (status.get("all_complete") is True and
            all(status.get("completed", {}).get(model) == 200 for model in ("qwen", "gpt")))


def valid_cpu_check(status):
    return (status.get("cpu_tests_passed") is True and status.get("tests") == CPU_TESTS and
            type(status.get("passed_count")) is int and status["passed_count"] >= len(CPU_TESTS) and
            status.get("submit_performed") is False and status.get("online_api_calls") == 0)


def atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_linked_roll32_dev_20261104":
        raise ValueError("Unexpected combined job directory")
    linked, rollout = (workspace / "query_cartridges_linked_selection_20261104",
                       workspace / "query_cartridges_rollout32_olmo_dev_20261104")
    completion_paths = (linked / "finished.json", rollout / "finished.json")
    if any(not path.is_file() for path in completion_paths):
        print("Waiting for both 200-case component studies to finish")
        return None
    if any(not finished_component(json.loads(path.read_text())) for path in completion_paths):
        raise ValueError("A component study did not finish 200 cases on both models")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.is_file():
            return json.loads(target.read_text())["job_id"]
        actual = json.loads((root / "manifest.json").read_text())
        expected = prepare_development_manifest(json.loads((rollout / "manifest.json").read_text()),
            json.loads((linked / "manifest.json").read_text()),
            json.loads((workspace / "query_cartridges_value_lr_20261104/olmo_profile.json").read_text()))
        validation = json.loads((root / "cpu_validation_passed.json").read_text())
        if actual != expected or not valid_cpu_check(validation):
            raise ValueError("The prepared manifest or CPU validation changed")
        if (root / "job.sbatch").read_text() != slurm_script(workspace, root) or \
           (root / "report.py").read_text() != report_stub(workspace, root):
            raise ValueError("Prepared execution or reporting scripts changed")
        with submission_slot() as available:
            if not available:
                print("Waiting for one slot under the 12-GPU limit")
                return None
            result = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j"],
                                    check=True, capture_output=True, text=True, timeout=20)
            jobs = [parts[0].strip() for line in result.stdout.splitlines()
                    if len(parts := line.split("|", 1)) == 2 and parts[1].strip() == "qc-linked-roll32"]
            if len(jobs) > 1:
                raise RuntimeError("More than one combined study exists")
            if jobs:
                job = jobs[0]
            else:
                sent = subprocess.run(["sbatch", "--parsable", "--comment=cartridge-linked-roll32",
                                       str(root / "job.sbatch")],
                                      check=True, capture_output=True, text=True, timeout=30)
                job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Invalid scheduler response")
            atomic(target, {"job_id": job, "gpus": 1, "models": actual["models"],
                "cases_per_model": 200, "source_disjoint_holdout": 800, "keep_fraction": .1,
                "student_rollout_tokens": 32, "selected_transfer_weight": 0, "online_api_calls": 0,
                "max_gpu_hours": 36, "max_hours_per_allocation": 6, "created": time.time()})
        return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--job-root", required=True, type=Path)
    args = parser.parse_args()
    job = submit(args.workspace, args.job_root)
    if job:
        from experiments.babilong_linked_selection.long_rollout_monitors import disable_timer, enable_timer
        enable_timer(args.workspace.resolve(), args.job_root.resolve(), "dev")
        disable_timer("submit")
        print("Combined study submitted:", job)


if __name__ == "__main__":
    main()
