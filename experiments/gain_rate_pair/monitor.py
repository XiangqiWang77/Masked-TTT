#!/usr/bin/env python3
"""Hourly paired reports and model-specific conditional holdout submissions."""
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
from experiments.gain_rate_pair.job import DEV, HOLDOUTS, batch_script, enable_monitor, report_stub, submit, validate_frozen
from experiments.gain_rate_pair.report import render
from experiments.gqa_head_gain.prepare import atomic
from experiments.gqa_head_gain.report import summarize


def model_view(report, model):
    """Extract one model, retaining model-specific errors and invalid cases."""
    if model not in report["models"]:
        raise ValueError("The model is missing from the development report")
    return dict(mode=report["mode"], expected=report["expected_per_model"], metric=report["metric"],
        overall=report["models"][model]["overall"], rows=report["models"][model]["rows"],
        errors=[item for item in report["errors"] if item.get("model") == model],
        invalid=[item for item in report["invalid"] if item.get("model") == model])


def development_decision(report):
    stats = report["overall"]
    if (report["mode"] != "development" or report["expected"] != 200 or stats["n"] != 200 or
        report["metric"] != "answer_focused_lexical_v2" or report["errors"] or report["invalid"] or
        stats["strictly_paired"]["n"] != 200):
        return dict(eligible=False, reason="development_incomplete_or_invalid")
    accuracy = stats["accuracy"]
    difference = {key: accuracy["cartridge"] - accuracy[key] for key in ("whole", "zero", "teacher")}
    difference["baseline"] = stats["strictly_paired"]["delta_pp"]
    eligible = all(value > 0 for value in difference.values())
    return dict(eligible=eligible, reason="beats_all_paired_development_controls" if eligible else
        "does_not_beat_all_paired_development_controls", delta_pp=difference,
        new_correct=stats["strictly_paired"]["new_correct"], new_wrong=stats["strictly_paired"]["new_wrong"])


def prepare_holdout(workspace, development, report, model):
    source = json.loads((development / "manifest.json").read_text())
    validate_partition(source["development_case_ids"], source["holdout_case_ids"])
    view = model_view(report, model)
    decision = development_decision(view)
    if not decision["eligible"]:
        atomic(development / ("no_eligible_" + model + ".json"), json.dumps(decision, indent=2) + "\n")
        return None, decision
    root = workspace / HOLDOUTS[model]
    root.mkdir(exist_ok=True)
    manifest = dict(source, mode="holdout", models=[model], case_count=800,
        case_ids=list(source["holdout_case_ids"]), scope=model + " paired original-SD gain-rate holdout",
        development_decision=decision, development_used_for_configuration_selection=True,
        holdout_evaluated_at_selection=False)
    with (root / "promotion.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "runtime").exists():
            if not (root / "manifest.json").is_file() or json.loads((root / "manifest.json").read_text()) != manifest:
                raise ValueError("The existing holdout has a different protocol")
        else:
            shutil.copytree(development / "runtime", root / "runtime",
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
            (root / "logs").mkdir(exist_ok=True)
            atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
            atomic(root / "development_snapshot.json", json.dumps(view, indent=2) + "\n")
            atomic(root / "report.py", report_stub(workspace, root))
            atomic(root / "job.sbatch", batch_script(root))
    return root, decision


def combine(manifest, development, holdout):
    model = manifest["models"][0]
    held = model_view(holdout, model)
    dev_rows, held_rows = development["rows"], held["rows"]
    if (development["metric"] != held["metric"] or development["errors"] or development["invalid"] or
        held["errors"] or held["invalid"] or len(dev_rows) != 200 or len(held_rows) != 800):
        raise ValueError("Complete valid 200/800 consistently scored cases are required")
    validate_partition([row["case"] for row in dev_rows], [row["case"] for row in held_rows])
    if ({row["case"] for row in dev_rows} != set(manifest["development_case_ids"]) or
        {row["case"] for row in held_rows} != set(manifest["holdout_case_ids"])):
        raise ValueError("The fixed case partition changed")
    rows, groups = dev_rows + held_rows, defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    viewed = set(manifest["qualitative_case_sources"])
    unviewed = [row for row in held_rows if row["case"].split("-", 2)[2] not in viewed]
    if len(unviewed) != 792:
        raise ValueError("Unexpected previously inspected source exclusion")
    return dict(mode="combined", expected_per_model=1000, metric=held["metric"],
        models={model: dict(overall=summarize(rows), groups={k:summarize(v) for k,v in groups.items()})},
        development=summarize(dev_rows), holdout=summarize(held_rows),
        holdout_excluding_two_previously_viewed_sources=summarize(unviewed),
        development_used_for_configuration_selection=True, prior_other_selection_cohort_analysis_seen=True)


def advance(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name not in (DEV, *HOLDOUTS.values()):
        raise ValueError("Unexpected gain-rate pair directory")
    subprocess.run([sys.executable, str(root / "report.py")], check=True,
                   capture_output=True, text=True, timeout=120)
    report = json.loads((root / "latest.json").read_text())
    if root.name == DEV:
        outcomes = {}
        for model in report["models"]:
            view = model_view(report, model)
            if view["overall"]["n"] < 200:
                outcomes[model] = dict(state="in_progress", n=view["overall"]["n"])
                continue
            held, decision = prepare_holdout(workspace, root, report, model)
            if held is None:
                outcomes[model] = dict(state="not_eligible", decision=decision)
                continue
            if not (held / "cpu_validation_passed.json").exists():
                validate_frozen(held, python)
            try:
                sent = submit(workspace, held)
            except RuntimeError as exc:
                if str(exc) == "The shared 12-GPU job cap is full":
                    outcomes[model] = dict(state="waiting_slot")
                    continue
                raise
            enable_monitor(workspace, held, python)
            outcomes[model] = dict(state="submitted", job_id=sent["job_id"])
        if (root / "finished.json").exists() and all(
            outcome["state"] in ("submitted", "not_eligible") for outcome in outcomes.values()):
            return dict(state="finished_development", models=outcomes)
        if (root / "finished.json").exists() and any(
            outcome["state"] == "in_progress" for outcome in outcomes.values()):
            return dict(state="terminal_without_complete_results", reason="invalid_or_missing_development_cases",
                        models=outcomes)
        if not (root / "finished.json").exists():
            sys.path.insert(0, str(workspace))
            from cartridge_job_watch import record_terminal
            terminal = record_terminal(root)
            if terminal:
                return dict(state="terminal_without_complete_results", scheduler_state=terminal, models=outcomes)
        return dict(state="in_progress", models=outcomes)
    if not (root / "finished.json").exists():
        sys.path.insert(0, str(workspace))
        from cartridge_job_watch import record_terminal
        terminal = record_terminal(root)
        return dict(state="terminal_without_complete_results" if terminal else "in_progress", scheduler_state=terminal)
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
    if result["state"] in ("terminal_without_complete_results", "finished_development", "complete"):
        subprocess.run(["systemctl", "--user", "stop", args.timer], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
