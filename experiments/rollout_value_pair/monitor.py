#!/usr/bin/env python3
"""Hourly paired report and one conditional source-disjoint holdout job."""
import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.prepare import atomic
from experiments.gqa_head_gain.report import summarize
from experiments.rollout_value_pair.job import DEV, HOLDOUT, batch_script, enable_monitor, report_stub, submit, validate_frozen
from experiments.rollout_value_pair.report import render
from experiments.self_study_prompt.study_gate import development_decision


def prepare_holdout(workspace, development, report):
    source = json.loads((development / "manifest.json").read_text())
    validate_partition(source["development_case_ids"], source["holdout_case_ids"])
    decision = development_decision(report)
    if not decision["eligible"]:
        atomic(development / "no_eligible_holdout.json", json.dumps(decision, indent=2) + "\n")
        return None, decision
    root = workspace / HOLDOUT
    root.mkdir(exist_ok=True)
    manifest = dict(source, mode="holdout", case_count=800, case_ids=list(source["holdout_case_ids"]),
        scope="Qwen paired original-SD rollout/Value rate, source-disjoint holdout",
        development_decision=decision, development_used_for_configuration_selection=True,
        holdout_evaluated_at_selection=False)
    with (root / "promotion.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "runtime").exists():
            if not (root / "manifest.json").exists() or json.loads((root / "manifest.json").read_text()) != manifest:
                raise ValueError("The existing holdout has a different protocol")
        else:
            shutil.copytree(development / "runtime", root / "runtime",
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
            (root / "logs").mkdir(exist_ok=True)
            atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
            atomic(root / "development_snapshot.json", json.dumps(report, indent=2) + "\n")
            atomic(root / "report.py", report_stub(workspace, root))
            atomic(root / "job.sbatch", batch_script(root))
    return root, decision


def combine(manifest, development, holdout):
    dev, held = development["rows"], holdout["rows"]
    if (development["metric"] != holdout["metric"] or development["errors"] or development["invalid"] or
        holdout["errors"] or holdout["invalid"] or len(dev) != 200 or len(held) != 800):
        raise ValueError("Complete valid 200/800 consistently scored cases are required")
    validate_partition([row["case"] for row in dev], [row["case"] for row in held])
    if ({row["case"] for row in dev} != set(manifest["development_case_ids"]) or
        {row["case"] for row in held} != set(manifest["holdout_case_ids"])):
        raise ValueError("The fixed case partition changed")
    rows, groups = dev + held, defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    viewed = set(manifest["qualitative_case_sources"])
    unviewed = [row for row in held if row["case"].split("-", 2)[2] not in viewed]
    if len(unviewed) != 792:
        raise ValueError("Unexpected previously inspected source exclusion")
    return dict(mode="combined", model="qwen", expected=1000, metric=holdout["metric"],
        overall=summarize(rows), development=summarize(dev), holdout=summarize(held),
        holdout_excluding_two_previously_viewed_sources=summarize(unviewed),
        groups={key: summarize(items) for key, items in sorted(groups.items())},
        development_used_for_configuration_selection=True, prior_other_selection_cohort_analysis_seen=True)


def advance(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name not in (DEV, HOLDOUT):
        raise ValueError("Unexpected paired-hyperparameter directory")
    subprocess.run([sys.executable, str(root / "report.py")], check=True,
                   capture_output=True, text=True, timeout=120)
    if not (root / "finished.json").exists():
        sys.path.insert(0, str(workspace))
        from cartridge_job_watch import record_terminal
        terminal = record_terminal(root)
        return dict(state="terminal_without_complete_results" if terminal else "in_progress", scheduler_state=terminal)
    report = json.loads((root / "latest.json").read_text())
    if root.name == DEV:
        held, decision = prepare_holdout(workspace, root, report)
        if held is None:
            return dict(state="not_eligible", decision=decision)
        if not (held / "cpu_validation_passed.json").exists():
            validate_frozen(held, python)
        try:
            sent = submit(workspace, held)
        except RuntimeError as exc:
            if str(exc) == "The shared 12-GPU job cap is full":
                return dict(state="waiting_slot")
            raise
        enable_monitor(workspace, held, python)
        return dict(state="submitted", job_id=sent["job_id"])
    manifest = json.loads((root / "manifest.json").read_text())
    development = json.loads((root / "development_snapshot.json").read_text())
    combined = combine(manifest, development, report)
    atomic(root / "combined_1000.json", json.dumps(combined, indent=2) + "\n")
    atomic(root / "combined_1000.md", render(combined) +
        "The development set selected the configuration. Holdout sources are disjoint; two had been "
        "previously inspected, and other selection variants on the cohort were previously analysed.\n")
    return dict(state="complete", report=str(root / "combined_1000.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "job-root", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timer", required=True)
    args = parser.parse_args()
    root = args.job_root.resolve()
    try:
        result = advance(args.workspace, root, args.python)
    except Exception as exc:
        atomic(root / "monitor_status.json", json.dumps(dict(state="monitor_error",
               error_type=type(exc).__name__, updated=time.time()), indent=2) + "\n")
        raise
    atomic(root / "monitor_status.json", json.dumps(dict(result, updated=time.time()), indent=2) + "\n")
    print(json.dumps(result, default=str))
    if result["state"] in ("terminal_without_complete_results", "not_eligible", "submitted", "complete"):
        subprocess.run(["systemctl", "--user", "stop", args.timer], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
