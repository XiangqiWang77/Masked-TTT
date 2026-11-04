#!/usr/bin/env python3
"""Refresh the paired development report at most once per hour."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.gqa_head_gain.prepare import atomic


def advance(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_prompt_aligned_dev_20261104":
        raise ValueError("Unexpected aligned-question experiment directory")
    subprocess.run([sys.executable, str(root / "report.py")], check=True,
                   capture_output=True, text=True, timeout=120)
    baseline = workspace / "query_cartridges_longsteps_200dev_20261104"
    if (root / "finished.json").exists() and (baseline / "finished.json").exists():
        report = json.loads((root / "latest.json").read_text())
        return dict(stage="both_runs_complete", paired={model: item["overall"]["n"]
                    for model, item in report["models"].items()}, invalid=len(report["invalid"]))
    if (root / "finished.json").exists():
        return dict(stage="waiting_for_baseline")
    sys.path.insert(0, str(workspace))
    from cartridge_job_watch import record_terminal
    terminal = record_terminal(root)
    return dict(stage="terminal_without_complete_results" if terminal else "in_progress",
                scheduler_state=terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--timer", required=True)
    args = parser.parse_args()
    root = args.job_root.resolve()
    try:
        result = advance(args.workspace, root)
    except Exception as exc:
        atomic(root / "monitor_status.json", json.dumps(dict(stage="monitor_error",
               error_type=type(exc).__name__, updated=time.time()), indent=2) + "\n")
        raise
    atomic(root / "monitor_status.json", json.dumps(dict(result, updated=time.time()), indent=2) + "\n")
    print(json.dumps(result))
    if result["stage"] in ("both_runs_complete", "terminal_without_complete_results"):
        subprocess.run(["systemctl", "--user", "stop", args.timer], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
