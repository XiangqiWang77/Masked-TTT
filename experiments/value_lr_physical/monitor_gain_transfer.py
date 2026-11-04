#!/usr/bin/env python3
"""Check one submitted diagnostic hourly; never submit or restart a job."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def confirmed_terminal(job):
    # A deployed monitor lives directly under the job directory; the shared
    # scheduler watcher is in its workspace parent. Tests import the package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from cartridge_job_watch import observed_terminal
    except ImportError:
        from experiments.value_lr_physical.cartridge_job_watch import observed_terminal
    return observed_terminal(job)


def scheduler_state(job):
    active = subprocess.run(["squeue", "-j", job, "-h", "-o", "%T|%R"],
                            check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    if active:
        state, reason = active.split("|", 1)
        return state, reason
    return confirmed_terminal(job) or "UNKNOWN", None


def monitor(root, schedule):
    job = str(json.loads((root / "submission.json").read_text())["job_id"])
    if not job.isdigit():
        raise ValueError("Invalid submitted job")
    state, reason = scheduler_state(job)
    if state in TERMINAL:
        sys.path.insert(0, str(root / "runtime/repo"))
        from experiments.value_lr_physical.report_gain_transfer import write_report
        write_report(root)
    report_path = root / "latest.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    now = datetime.now(timezone.utc)
    result = dict(job_id=job, checked_utc=now.isoformat(), state=state, reason=reason,
                  complete={model: info["overall"]["cases"] for model, info in report["models"].items()}
                  if report else {"qwen": 0, "gpt": 0})
    atomic(root / "monitor.json", result)
    if schedule.exists():
        data = json.loads(schedule.read_text())
        data.update(gain_transfer_job_id=job, gain_transfer_last_check_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    gain_transfer_next_check_not_before_utc=(now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    gain_transfer_last_state=state)
        atomic(schedule, data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    result = monitor(args.job_root, args.schedule)
    print(json.dumps(result))
    if result["state"] in TERMINAL:
        subprocess.run(["systemctl", "--user", "disable", "--now", "query-cartridges-gain-transfer.timer"],
                       check=False, capture_output=True, text=True)


if __name__ == "__main__":
    main()
