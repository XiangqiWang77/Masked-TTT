#!/usr/bin/env python3
"""Refresh the combined study's report at most once per timer invocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.babilong_linked_selection.long_rollout_monitors import disable_timer


def done(report):
    return bool(report["models"]) and all(
        item["decision"]["reason"] != "development_or_matched_baseline_incomplete"
        for item in report["models"].values()
    )


def refresh(workspace, root, kind):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or kind not in ("dev", "holdout"):
        raise ValueError("Unexpected monitor scope")
    subprocess.run([sys.executable, str(root / "report.py")], check=True, timeout=180)
    report = json.loads((root / "latest.json").read_text())
    if kind == "holdout" and (root / "finished.json").exists():
        from experiments.babilong_linked_selection.long_rollout_report import combine_reports, write_report
        development = workspace / "query_cartridges_linked_roll32_dev_20261104/latest.json"
        if development.is_file():
            combined = combine_reports(json.loads(development.read_text()), report)
            if combined is not None:
                write_report(combined, root / "combined1000")
    if (root / "finished.json").exists() and done(report):
        if kind == "dev":
            from experiments.babilong_linked_selection.long_rollout_holdout import submit
            outcome = submit(workspace, root)
            if outcome["state"] not in ("submitted", "not_eligible"):
                return outcome["state"]
            disable_timer(kind)
            return "development_" + outcome["state"]
        disable_timer(kind)
        return "finished"
    if not (root / "finished.json").exists():
        sys.path.insert(0, str(workspace))
        from cartridge_job_watch import record_terminal
        if record_terminal(root) is not None:
            disable_timer(kind)
            return "terminal"
    return "waiting"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("dev", "holdout"))
    args = parser.parse_args()
    print("Combined report:", refresh(args.workspace, args.job_root, args.kind))


if __name__ == "__main__":
    main()
