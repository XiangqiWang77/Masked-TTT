#!/usr/bin/env python3
"""Hourly report and source-disjoint promotion for the query-head candidate."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.prepare import atomic, report_stub, slurm_script, submit
from experiments.gqa_head_gain.report import summarize

INSPECTED = {"qa1-300023", "qa2-300037"}
TIMER = "query-cartridges-gqa-report.timer"


def monitor_units(workspace, dev_root):
    entry = dev_root / "runtime/repo/experiments/gqa_head_gain/monitor.py"
    if any(char.isspace() for char in str(workspace) + str(dev_root)):
        raise ValueError("Systemd monitor paths cannot contain whitespace")
    service = ("[Unit]\nDescription=Report query-head gain development and holdout\n\n"
               "[Service]\nType=oneshot\n"
               "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
               f"ExecStart=/usr/bin/python3 {entry} --workspace {workspace} --dev-root {dev_root}\n")
    timer = ("[Unit]\nDescription=Check query-head gain experiment once per hour\n\n"
             "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
             "[Install]\nWantedBy=timers.target\n")
    return service, timer


def enable_monitor(workspace, dev_root):
    unit_dir = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    service, timer = monitor_units(workspace, dev_root)
    atomic(unit_dir / "query-cartridges-gqa-report.service", service)
    atomic(unit_dir / TIMER, timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", TIMER],
                   check=True, capture_output=True, text=True)


def promoted_models(manifest, report):
    if (report["expected_per_model"] != 200 or report["metric"] != "answer_focused_lexical_v2" or
        report["gain_head_mode"] != "query" or not report["cached_scores"] or
        report["online_api_calls"] != 0 or not report["evaluation_answers_used_only_posthoc"] or
        report["errors"] or report["invalid"]):
        raise ValueError("A clean complete development report is required")
    selected = []
    for model in manifest["models"]:
        item = report["models"][model]
        rows = item["rows"]
        if len(rows) != 200 or {row["case"] for row in rows} != set(manifest["case_ids"]):
            raise RuntimeError("Development cases are incomplete: " + model)
        expected = summarize(rows)
        if item["overall"] != expected:
            raise ValueError("Stored development metrics do not match per-case scores")
        values = expected["accuracy"]
        eligible = all(values["cartridge"] > values[name] for name in ("teacher", "zero", "whole"))
        if eligible != item["development_gate"]["eligible"]:
            raise ValueError("Stored development gate does not match the case scores")
        if eligible:
            selected.append(model)
    return selected


def prepare_holdout(workspace, dev_root, models):
    root = workspace / "query_cartridges_gqa_gain_holdout_20261104"
    dev = json.loads((dev_root / "manifest.json").read_text())
    validate_partition(dev["development_case_ids"], dev["holdout_case_ids"])
    if not models or any(model not in dev["models"] for model in models):
        raise ValueError("At least one development-selected model is required")
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists():
            previous = json.loads((root / "manifest.json").read_text())
            if previous["models"] != models:
                raise ValueError("The submitted holdout uses a different model selection")
            return root
        if (root / "cases").exists():
            raise ValueError("Do not replace evaluated holdout data")
        runtime = root / "runtime"
        if runtime.exists():
            shutil.rmtree(runtime)
        shutil.copytree(dev_root / "runtime", runtime,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        manifest = dict(dev, scope="Source-disjoint 800 cases after query-head development",
            mode="holdout", case_count=800, case_ids=list(dev["holdout_case_ids"]), models=list(models),
            development_results_seen=True, holdout_evaluated_at_selection=False)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "job.sbatch", slurm_script(root))
        atomic(root / "report.py", report_stub(workspace, root,
            workspace / "query_cartridges_linked_holdout_20261104"))
        shutil.copy2(dev_root / "cpu_validation_passed.json", root / "cpu_validation_passed.json")
        (root / "logs").mkdir(exist_ok=True)
    return root


def combine(development, holdout):
    if (development["expected_per_model"] != 200 or holdout["expected_per_model"] != 800 or
        development["metric"] != holdout["metric"] or development["gain_head_mode"] != "query" or
        holdout["gain_head_mode"] != "query" or development["errors"] or development["invalid"] or
        holdout["errors"] or holdout["invalid"]):
        raise ValueError("Development and holdout results are incompatible")
    models = {}
    for model, held in holdout["models"].items():
        dev_rows, held_rows = development["models"][model]["rows"], held["rows"]
        if len(dev_rows) != 200 or len(held_rows) != 800:
            continue
        validate_partition([row["case"] for row in dev_rows], [row["case"] for row in held_rows])
        rows = dev_rows + held_rows
        groups = defaultdict(list)
        for row in rows:
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        strict = [row for row in held_rows if row["case"].split("-", 2)[2] not in INSPECTED]
        stats = summarize(strict)
        deltas = {name: stats["versus_" + name]["delta_pp"] for name in ("teacher", "zero", "whole")}
        passed = len(strict) == 792 and deltas["teacher"] >= 3 - 1e-9 and deltas["zero"] > 0 and deltas["whole"] > 0
        models[model] = dict(overall=summarize(rows), development=summarize(dev_rows), holdout=summarize(held_rows),
            holdout_excluding_viewed_examples=stats, holdout_target_met=passed,
            groups={key: summarize(value) for key, value in groups.items()}, rows=rows)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), metric=development["metric"],
                gain_head_mode="query", expected_per_model=1000, models=models)


def render_combined(report):
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    lines = ["# Query-head gain: BABILONG 1,000", "",
        "| Model / group | n | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model, data in report["models"].items():
        groups = [(model, data["overall"])]+[(model + " " + task, data["groups"]["task:" + task])
                  for task in ("qa" + str(i) for i in range(1, 11)) if "task:" + task in data["groups"]]
        for name, group in groups:
            acc = group["accuracy"]
            lines.append(f"| {name} | {group['n']} | {fmt(acc['whole'])} | {fmt(acc['zero'])} | "
                         f"{fmt(acc['teacher'])} | {fmt(acc['cartridge'])} | "
                         f"{fmt(group['versus_teacher']['delta_pp'])} |")
        lines.append(f"\n{model}: 792-case source-disjoint target met: {data['holdout_target_met']}.\n")
    lines += ["The 200-case development split chooses the configuration; 800 other source-disjoint cases "
              "are then evaluated. Two previously inspected source questions are excluded from the 792-case "
              "target calculation. Aggregate earlier BABILONG results were viewed, so this is not a fully blind holdout.", ""]
    return "\n".join(lines)


def monitor(workspace, dev_root):
    workspace, dev_root = workspace.resolve(), dev_root.resolve()
    if dev_root.parent != workspace or dev_root.name != "query_cartridges_gqa_gain_dev_20261104":
        raise ValueError("Unexpected development root")
    with (dev_root / "monitor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run([sys.executable, str(dev_root / "report.py")], check=True, timeout=120)
        if not (dev_root / "finished.json").exists():
            return "development_in_progress"
        manifest = json.loads((dev_root / "manifest.json").read_text())
        development = json.loads((dev_root / "latest.json").read_text())
        models = promoted_models(manifest, development)
        if not models:
            atomic(dev_root / "development_decision.json", json.dumps(
                dict(models_promoted=[], reason="No model beat its same-run controls"), indent=2) + "\n")
            subprocess.run(["systemctl", "--user", "disable", "--now", TIMER],
                           check=True, capture_output=True, text=True)
            return "no_model_promoted"
        hold_root = prepare_holdout(workspace, dev_root, models)
        submitted = submit(workspace, hold_root)
        subprocess.run([sys.executable, str(hold_root / "report.py")], check=True, timeout=120)
        if not (hold_root / "finished.json").exists():
            return "holdout_in_progress_" + submitted["job_id"]
        holdout = json.loads((hold_root / "latest.json").read_text())
        report = combine(development, holdout)
        atomic(dev_root / "combined.json", json.dumps(report, indent=2) + "\n")
        atomic(dev_root / "combined.md", render_combined(report))
        subprocess.run(["systemctl", "--user", "disable", "--now", TIMER],
                       check=True, capture_output=True, text=True)
        return "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--dev-root", required=True, type=Path)
    args = parser.parse_args()
    print(monitor(args.workspace, args.dev_root))


if __name__ == "__main__":
    main()
