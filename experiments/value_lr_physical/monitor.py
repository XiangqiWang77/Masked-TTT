#!/usr/bin/env python3
"""Refresh one physical Value-LR holdout once an hour and combine on completion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.gqa_head_gain.prepare import atomic
from experiments.value_lr_physical.report import combine, write


def advance(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name not in (
        "query_cartridges_value_lr_physical_gpt_holdout_20261104",
        "query_cartridges_value_lr_physical_qwen_holdout_20261104"):
        raise ValueError("Unexpected one-model physical holdout directory")
    subprocess.run([sys.executable, str(root / "report.py")], check=True,
                   capture_output=True, text=True, timeout=120)
    if (root / "finished.json").exists():
        development_manifest = json.loads((root / "development_manifest.json").read_text())
        development = json.loads((root / "development_snapshot.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        holdout = json.loads((root / "latest.json").read_text())
        result = combine(development_manifest, development, manifest, holdout)
        if result is None:
            raise ValueError("The completed holdout does not have 800 valid scored cases")
        write(result, root / "combined_1000")
        return dict(stage="complete", report=str(root / "combined_1000.md"))
    sys.path.insert(0, str(workspace))
    from cartridge_job_watch import record_terminal
    terminal = record_terminal(root)
    return dict(stage="terminal_without_complete_results" if terminal else "in_progress",
                scheduler_state=terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--timer", required=True)
    args = parser.parse_args()
    root = args.job_root.resolve()
    try:
        result = advance(args.workspace, root)
    except Exception as exc:
        atomic(root / "monitor_status.json", json.dumps(
            dict(stage="monitor_error", error_type=type(exc).__name__, updated=time.time()), indent=2) + "\n")
        raise
    atomic(root / "monitor_status.json", json.dumps(dict(result, updated=time.time()), indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["stage"] in ("complete", "terminal_without_complete_results"):
        subprocess.run(["systemctl", "--user", "stop", args.timer],
                       check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
