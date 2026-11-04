#!/usr/bin/env python3
"""Start the frozen 200-case comparison only after the source-only probe succeeds."""
import argparse
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from experiments.gqa_head_gain.prepare import atomic
from experiments.self_study_prompt.study_gate import probe_decision
from experiments.self_study_prompt.study_job import DEV, PROBE, enable_monitor, submit


def advance(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != DEV:
        raise ValueError("Unexpected paired-study development directory")
    probe = workspace / PROBE
    if not probe.is_file():
        return dict(state="waiting_for_prompt_probe")
    decision = probe_decision(json.loads(probe.read_text()))
    if not decision["eligible"]:
        return dict(state="probe_not_eligible", decision=decision)
    try:
        sent = submit(workspace, root)
    except RuntimeError as exc:
        if str(exc) == "The shared 12-GPU job cap is full":
            return dict(state="waiting_slot", decision=decision)
        raise
    marker = root / "monitor_registered.json"
    if marker.exists():
        if json.loads(marker.read_text()).get("job_id") != sent["job_id"]:
            raise ValueError("A different paired-study job owns the monitor")
    else:
        enable_monitor(workspace, root, python)
        atomic(marker, json.dumps(dict(job_id=sent["job_id"], interval_seconds=3600), indent=2) + "\n")
    return dict(state="submitted", job_id=sent["job_id"], decision=decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "job-root", "python"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        result = advance(args.workspace, args.job_root, args.python)
    except Exception as exc:
        atomic(args.job_root / "trigger_status.json", json.dumps(dict(state="error",
               error_type=type(exc).__name__, updated=time.time()), indent=2) + "\n")
        raise
    atomic(args.job_root / "trigger_status.json", json.dumps(dict(result, updated=time.time()), indent=2) + "\n")
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
