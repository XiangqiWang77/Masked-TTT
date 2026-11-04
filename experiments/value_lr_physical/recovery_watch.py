#!/usr/bin/env python3
"""Create recovery reports from the existing hourly physical-job reports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.value_lr_physical.recovery_report import collect, write


def advance(workspace):
    workspace = workspace.resolve()
    states = {}
    for model in ("qwen", "gpt"):
        root = workspace / ("query_cartridges_value_lr_physical_" + model + "_holdout_20261104")
        source = root / "combined_1000.json"
        if not source.exists():
            source = root / "latest.json"
        item = dict(stage="waiting_for_report", terminal=False)
        if source.exists():
            result = collect(json.loads(source.read_text()))
            write(result, root / "recovery_latest")
            item.update(stage="reported", n=result["overall"]["n"], expected=result["expected"],
                        report=str(root / "recovery_latest.md"))
        status = root / "monitor_status.json"
        if status.exists():
            monitor = json.loads(status.read_text())
            if monitor.get("stage") in ("complete", "terminal_without_complete_results"):
                item.update(terminal=True, monitor_stage=monitor["stage"])
        elif model == "qwen":
            launcher = workspace / "query_cartridges_value_lr_physical_qwen_launcher_status.json"
            if launcher.exists() and json.loads(launcher.read_text()).get("stage") == "no_eligible_model":
                item.update(terminal=True, monitor_stage="no_eligible_model")
        states[model] = item
    return dict(stage="all_terminal" if all(item["terminal"] for item in states.values()) else "waiting",
                models=states, updated=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--timer", required=True)
    args = parser.parse_args()
    result = advance(args.workspace)
    path = args.workspace / "query_cartridges_recovery_watch_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(result, indent=2))
    if result["stage"] == "all_terminal":
        subprocess.run(["systemctl", "--user", "stop", args.timer],
                       check=True, capture_output=True, text=True, timeout=20)


if __name__ == "__main__":
    main()
