#!/usr/bin/env python3
"""Hourly, promote one completed model's predeclared Value-LR winner."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.gqa_head_gain.prepare import atomic
from experiments.value_lr_physical.prepare import prepare, submit
from experiments.value_lr_physical.protocol import development_choice


def ensure_report_timer(workspace, root, model):
    name = "query-cartridges-value-physical-" + model + "-report"
    status = subprocess.run(["systemctl", "--user", "is-active", name + ".timer"],
                            capture_output=True, text=True, timeout=20, check=False)
    if status.returncode == 0 and status.stdout.strip() == "active":
        return name + ".timer"
    script = root / "runtime/repo/experiments/value_lr_physical/monitor.py"
    command = ["systemd-run", "--user", "--unit=" + name, "--on-active=1h", "--on-unit-active=1h",
        "--timer-property=AccuracySec=1m", "--setenv=PATH=/opt/slurm/current/bin:/usr/local/bin:/usr/bin:/bin",
        "--setenv=OPENAI_API_KEY=", "--setenv=OPENAI_API_KEY_FILE=", "/usr/bin/python3", str(script),
        "--workspace", str(workspace), "--job-root", str(root), "--timer", name + ".timer"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return name + ".timer"


def advance(workspace, python, model):
    workspace = workspace.resolve()
    if model not in ("qwen", "gpt"):
        raise ValueError("Unsupported model")
    dev = workspace / "query_cartridges_value_lr_20261104"
    latest = json.loads((dev / "latest.json").read_text())
    item = latest["models"][model]
    if item["overall"]["n"] < 200:
        return dict(stage="development_in_progress", model=model, cases=item["overall"]["n"])
    development = json.loads((dev / "manifest.json").read_text())
    choice = development_choice(development, latest, model)
    if choice["value_learning_rate"] is None:
        return dict(stage="no_eligible_model", model=model, choice=choice)
    root = workspace / ("query_cartridges_value_lr_physical_" + model + "_holdout_20261104")
    prepare(workspace, root, python, model)
    job = submit(workspace, root)
    timer = ensure_report_timer(workspace, root, model)
    return dict(stage="submitted", model=model, job_id=job["job_id"],
                fixed_value_learning_rate=job["fixed_value_learning_rate"], report_timer=timer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--model", choices=("qwen", "gpt"), required=True)
    parser.add_argument("--timer", required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    prefix = "query_cartridges_value_lr_physical_" + args.model + "_launcher"
    with (workspace / (prefix + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            result = advance(workspace, args.python, args.model)
        except Exception as exc:
            atomic(workspace / (prefix + "_status.json"), json.dumps(
                dict(stage="launcher_error", error_type=type(exc).__name__, updated=time.time()), indent=2) + "\n")
            raise
        atomic(workspace / (prefix + "_status.json"), json.dumps(dict(result, updated=time.time()), indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if result["stage"] in ("submitted", "no_eligible_model"):
            subprocess.run(["systemctl", "--user", "stop", args.timer],
                           check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
