#!/usr/bin/env python3
"""Hourly report and conditional source-disjoint stronger-Value holdout."""
import argparse
from collections import defaultdict
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.long_rollout_protocol import summarize, validate_partition
from experiments.gqa_head_gain.prepare import atomic, slurm_script
from experiments.long_rollout_steps.extend import isolate_gpu_preflight
from experiments.value_lr_physical.strong_dev_prepare import CPU_TESTS, report_stub
from experiments.value_lr_physical.strong_dev_report import render

DEV = "query_cartridges_strong_value_gpt_dev_20261104"
HOLDOUT = "query_cartridges_strong_value_gpt_holdout_20261104"
HOLDOUT_TIMER = "query-cartridges-strong-value-holdout.timer"


def development_decision(report):
    stats, pair = report["overall"], report["versus_fixed_baseline"]
    if (report["mode"] != "development" or report["expected"] != 200 or stats["n"] != 200 or
        report["errors"] or report["invalid"]):
        return dict(eligible=False, reason="development_incomplete_or_invalid")
    accuracy = stats["accuracy"]
    better = all(accuracy["cartridge"] > accuracy[baseline] for baseline in ("whole", "zero", "teacher"))
    return dict(eligible=better, reason="beats_all_controls" if better else "does_not_beat_all_controls",
                accuracy=accuracy, matched_baseline=pair)


def holdout_script(root):
    script = slurm_script(root)
    changes = {
        "qc-gqa-gain-hold": "qc-strong-value-hold",
        "gqa_head_gain/execute.py": "value_lr_physical/strong_dev_execute.py",
        "#SBATCH --partition=gpu,gpu_h100,gpu_h200,gpu_b200,gpu_rtx6000,scavenge_gpu":
            "#SBATCH --partition=gpu_h200,gpu_h100,gpu_b200,gpu_rtx6000,scavenge_gpu",
        "#SBATCH --constraint=gpu:a40|gpu:l40s|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell":
            "#SBATCH --constraint=gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell",
        "#SBATCH --time-min=00:30:00": "#SBATCH --time-min=00:20:00",
    }
    for old, new in changes.items():
        if script.count(old) != 1:
            raise ValueError("Unexpected stronger-Value holdout script")
        script = script.replace(old, new, 1)
    return isolate_gpu_preflight(script)


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
        scope="GPT-OSS original-SD 0.003 Value rate, source-disjoint holdout",
        development_decision=decision, development_used_for_configuration_selection=True,
        holdout_evaluated_at_selection=False)
    with (root / "promotion.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").is_file():
            if json.loads((root / "manifest.json").read_text()) != manifest:
                raise ValueError("The existing holdout uses a different configuration")
            return root, decision
        runtime = root / "runtime"
        if not runtime.exists():
            shutil.copytree(development / "runtime", runtime,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
            (root / "logs").mkdir(exist_ok=True)
            atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
            atomic(root / "development_snapshot.json", json.dumps(report, indent=2) + "\n")
            atomic(root / "report.py", report_stub(workspace, root))
            atomic(root / "job.sbatch", holdout_script(root))
        elif (not (root / "manifest.json").is_file() or
              json.loads((root / "manifest.json").read_text()) != manifest):
            raise ValueError("The incomplete holdout runtime uses a different protocol")
    return root, decision


def enable_holdout_timer(workspace, root, python):
    unit = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    service = ("[Unit]\nDescription=Report original-SD stronger-Value GPT-OSS holdout\n\n"
        "[Service]\nType=oneshot\n"
        "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
        f"ExecStart=/usr/bin/python3 {root}/runtime/repo/experiments/value_lr_physical/strong_dev_monitor.py "
        f"--workspace {workspace} --job-root {root} --python {python} --timer {HOLDOUT_TIMER}\n")
    timer = ("[Unit]\nDescription=Update stronger-Value holdout once per hour\n\n"
        "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
        "[Install]\nWantedBy=timers.target\n")
    atomic(unit / "query-cartridges-strong-value-holdout.service", service)
    atomic(unit / HOLDOUT_TIMER, timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", HOLDOUT_TIMER],
                   check=True, capture_output=True, text=True)


def submit_holdout(workspace, root, python):
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            job = json.loads(target.read_text())["job_id"]
        else:
            frozen = root / "runtime/repo"
            env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            check = subprocess.run([str(python), "-m", "pytest", "--rootdir", str(frozen),
                "--confcutdir", str(frozen), "-q", *CPU_TESTS], cwd=frozen, env=env,
                text=True, capture_output=True, timeout=180)
            matched = re.search(r"\b(\d+) passed\b", check.stdout)
            if check.returncode or not matched:
                raise RuntimeError("Frozen stronger-Value holdout tests failed:\n" + check.stdout + check.stderr)
            atomic(root / "cpu_validation_passed.json", json.dumps(dict(cpu_tests_passed=True,
                   passed_count=int(matched.group(1)), gpu_preflight_required=True, online_api_calls=0), indent=2) + "\n")
            with submission_slot() as available:
                if not available:
                    return dict(state="waiting_slot")
                queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j"],
                                       check=True, capture_output=True, text=True, timeout=20)
                jobs = [parts[0].strip() for line in queue.stdout.splitlines()
                        if len(parts := line.split("|", 1)) == 2 and parts[1].strip() == "qc-strong-value-hold"]
                if len(jobs) > 1:
                    raise RuntimeError("More than one stronger-Value holdout job exists")
                if jobs:
                    job = jobs[0]
                else:
                    sent = subprocess.run(["sbatch", "--parsable", "--comment=original-sd-strong-value-holdout",
                        str(root / "job.sbatch")], check=True, capture_output=True, text=True, timeout=30)
                    job = sent.stdout.strip().split(";", 1)[0]
                if not job.isdigit():
                    raise ValueError("Unexpected scheduler response")
                atomic(target, json.dumps(dict(job_id=job, gpus=1, model="gpt", cases=800,
                    value_learning_rate=.003, online_api_calls=0, created=time.time()), indent=2) + "\n")
    enable_holdout_timer(workspace, root, python)
    return dict(state="submitted", job_id=job)


def combine(manifest, dev, holdout):
    development, held = dev["rows"], holdout["rows"]
    if (dev["metric"] != holdout["metric"] or dev["errors"] or dev["invalid"] or
        holdout["errors"] or holdout["invalid"]):
        raise ValueError("Cannot combine invalid or differently scored results")
    if len(development) != 200 or len(held) != 800:
        return None
    validate_partition([row["case"] for row in development], [row["case"] for row in held])
    if ({row["case"] for row in development} != set(manifest["development_case_ids"]) or
        {row["case"] for row in held} != set(manifest["holdout_case_ids"])):
        raise ValueError("The combined case identities differ from the fixed split")
    groups = defaultdict(list)
    for row in development + held:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    return dict(mode="combined", model="gpt", expected=1000, fixed_value_learning_rate=.003,
        metric=holdout["metric"], overall=summarize(development + held), development=summarize(development),
        holdout=summarize(held), versus_fixed_baseline=dev["versus_fixed_baseline"],
        groups={name: summarize(items) for name, items in groups.items()},
        development_used_for_configuration_selection=True, previously_viewed_holdout_sources=2)


def advance(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name not in (DEV, HOLDOUT):
        raise ValueError("Unexpected stronger-Value job directory")
    subprocess.run([sys.executable, str(root / "report.py")], check=True,
                   capture_output=True, text=True, timeout=120)
    if not (root / "finished.json").exists():
        sys.path.insert(0, str(workspace))
        from cartridge_job_watch import record_terminal
        terminal = record_terminal(root)
        return dict(state="terminal_without_complete_results" if terminal else "in_progress",
                    scheduler_state=terminal)
    report = json.loads((root / "latest.json").read_text())
    if root.name == DEV:
        held, decision = prepare_holdout(workspace, root, report)
        if held is None:
            return dict(state="not_eligible", decision=decision)
        return submit_holdout(workspace, held, python)
    manifest = json.loads((root / "manifest.json").read_text())
    development = json.loads((root / "development_snapshot.json").read_text())
    combined = combine(manifest, development, report)
    if combined is None:
        raise ValueError("The completed holdout has fewer than 800 valid cases")
    atomic(root / "combined_1000.json", json.dumps(combined, indent=2) + "\n")
    atomic(root / "combined_1000.md", render(combined) +
           "The combined result includes 200 development cases used for selection and 800 source-disjoint "
           "holdout cases; two holdout source questions had been inspected previously.\n")
    return dict(state="complete", report=str(root / "combined_1000.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
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
