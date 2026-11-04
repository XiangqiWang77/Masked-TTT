#!/usr/bin/env python3
"""Promote aligned-question SD to one 800-case-per-model job only if effective."""
import argparse
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
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.prepare import atomic, slurm_script
from experiments.long_rollout_steps.extend import isolate_gpu_preflight
from experiments.prompt_alignment.prepare import CPU_TESTS

HOLDOUT = "query_cartridges_prompt_aligned_holdout_20261104"
TIMER = "query-cartridges-prompt-aligned-holdout-report.timer"


def choose_models(manifest, report):
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    if (manifest["mode"] != "development" or manifest["case_count"] != 200 or
        set(manifest["case_ids"]) != set(manifest["development_case_ids"]) or
        manifest["align_student_teacher_prompt"] is not True or
        manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        report["metric"] != "answer_focused_lexical_v2" or report["expected_per_model"] != 200):
        raise ValueError("The aligned-question development protocol changed")
    selected, decisions = [], {}
    for model in manifest["models"]:
        item = report["models"][model]
        stats, rows = item["candidate_all"], item["candidate_rows"]
        model_errors = [x for x in report["errors"] + report["invalid"] if x["model"] == model]
        complete = (item["candidate_completed"] == 200 and stats["n"] == 200 and
                    len({row["case"] for row in rows}) == 200 and
                    {row["case"] for row in rows} == set(manifest["development_case_ids"]) and not model_errors)
        accuracy = stats["accuracy"]
        if not complete:
            decisions[model] = dict(eligible=False, reason="incomplete_or_invalid")
            continue
        comparisons = ("whole", "zero", "teacher")
        eligible = all(accuracy["aligned"] > accuracy[key] for key in comparisons)
        decisions[model] = dict(eligible=eligible, accuracy=accuracy,
            matched_original_sd=item["overall"]["aligned_vs_baseline"],
            reason="beats_all_controls" if eligible else "does_not_beat_all_controls")
        if eligible:
            selected.append(model)
    return selected, decisions


def holdout_manifest(development, models, decisions):
    return dict(development, mode="holdout", scope="Preselected aligned-question SD source-disjoint holdout",
                case_count=800, case_ids=list(development["holdout_case_ids"]), models=models,
                development_decisions=decisions, development_used_for_configuration_selection=True,
                holdout_used_for_configuration_selection=False, max_gpu_hours=36)


def report_stub(workspace, root):
    refs = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    args = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
            "--references", str(refs), "--scorer-dir", str(root / "runtime/scorer"), "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport fcntl\nfrom pathlib import Path\nimport sys\n"
        f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
        "from experiments.prompt_alignment.holdout_report import main\n"
        f"sys.argv = [sys.argv[0], *{args!r}]\n"
        f"with Path({str(root / 'report.lock')!r}).open('a+') as lock:\n"
        "    fcntl.flock(lock, fcntl.LOCK_EX)\n    main()\n")


def holdout_script(root):
    script = slurm_script(root)
    for old, new in (("qc-gqa-gain-hold", "qc-prompt-align-hold"),
                     ("gqa_head_gain/execute.py", "prompt_alignment/execute.py")):
        if script.count(old) != 1:
            raise ValueError("Unexpected holdout batch script")
        script = script.replace(old, new, 1)
    return isolate_gpu_preflight(script)


def freeze(workspace):
    workspace = workspace.resolve()
    development = workspace / "query_cartridges_prompt_aligned_dev_20261104"
    if not (development / "finished.json").is_file():
        sys.path.insert(0, str(workspace))
        from cartridge_job_watch import observed_terminal
        if (development / "submission.json").is_file():
            job = str(json.loads((development / "submission.json").read_text())["job_id"])
            state = observed_terminal(job)
            if state is not None:
                return dict(state="prerequisite_terminal", prerequisite="candidate",
                            job_id=job, scheduler_state=state)
        return dict(state="waiting")
    # Re-score completed answers under the same frozen scorer immediately before promotion.
    subprocess.run([sys.executable, str(development / "report.py")], check=True,
                   capture_output=True, text=True, timeout=120)
    source = json.loads((development / "manifest.json").read_text())
    report = json.loads((development / "latest.json").read_text())
    models, decisions = choose_models(source, report)
    root = workspace / HOLDOUT
    root.mkdir(exist_ok=True)
    with (root / "promotion.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").is_file():
            manifest = json.loads((root / "manifest.json").read_text())
            return dict(state="prepared", root=root, manifest=manifest)
        if not models:
            atomic(root / "no_eligible_model.json", json.dumps(dict(decisions=decisions), indent=2) + "\n")
            return dict(state="not_eligible", decisions=decisions)
        manifest = holdout_manifest(source, models, decisions)
        source_runtime, runtime = development / "runtime", root / "runtime"
        if not runtime.exists():
            shutil.copytree(source_runtime, runtime, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
            # The update/inference runtime is copied unchanged. These files
            # only manage, score, or monitor the post-development holdout.
            for file in ("holdout_report.py", "holdout_monitor.py", "promote.py"):
                shutil.copy2(REPO / "experiments/prompt_alignment" / file,
                             runtime / "repo/experiments/prompt_alignment" / file)
            shutil.copy2(REPO / "tests/test_prompt_alignment_holdout.py", runtime / "repo/tests/")
            (root / "logs").mkdir(exist_ok=True)
            atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
            atomic(root / "development_snapshot.json", json.dumps(report, indent=2) + "\n")
            atomic(root / "report.py", report_stub(workspace, root))
            atomic(root / "job.sbatch", holdout_script(root))
        elif (not (root / "manifest.json").exists() or
              json.loads((root / "manifest.json").read_text()) != manifest):
            raise ValueError("Existing holdout runtime has a different protocol")
        return dict(state="prepared", root=root, manifest=manifest)


def validate_cpu(root, python):
    frozen = root / "runtime/repo"
    env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    tests = CPU_TESTS + ["tests/test_prompt_alignment_holdout.py"]
    result = subprocess.run([str(python), "-m", "pytest", "--rootdir", str(frozen),
        "--confcutdir", str(frozen), "-q", *tests], cwd=frozen, env=env,
        text=True, capture_output=True, timeout=180)
    match = re.search(r"\b(\d+) passed\b", result.stdout)
    if result.returncode or not match:
        raise RuntimeError("Frozen holdout CPU validation failed:\n" + result.stdout + result.stderr)
    data = dict(cpu_tests_passed=True, passed_count=int(match.group(1)), tests=tests,
                gpu_preflight_required=True, online_api_calls=0)
    atomic(root / "cpu_validation_passed.json", json.dumps(data, indent=2) + "\n")
    return data


def enable_holdout_timer(workspace, root):
    unit = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    service = ("[Unit]\nDescription=Report aligned-question SD holdout\n\n[Service]\nType=oneshot\n"
        "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
        f"ExecStart=/usr/bin/python3 {root}/runtime/repo/experiments/prompt_alignment/holdout_monitor.py "
        f"--workspace {workspace} --job-root {root} --timer {TIMER}\n")
    timer = ("[Unit]\nDescription=Update aligned-question SD holdout once per hour\n\n"
             "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
             "[Install]\nWantedBy=timers.target\n")
    atomic(unit / "query-cartridges-prompt-aligned-holdout-report.service", service)
    atomic(unit / TIMER, timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", TIMER], check=True, capture_output=True, text=True)


def promote(workspace, python):
    workspace = workspace.resolve()
    prepared = freeze(workspace)
    if prepared["state"] != "prepared":
        return prepared
    root, manifest = prepared["root"], prepared["manifest"]
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            job = json.loads(target.read_text())["job_id"]
        else:
            validate_cpu(root, python)
            with submission_slot() as available:
                if not available:
                    return dict(state="waiting_slot")
                running = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j"],
                                         check=True, capture_output=True, text=True, timeout=20)
                jobs = [parts[0].strip() for line in running.stdout.splitlines()
                        if len(parts := line.split("|", 1)) == 2 and parts[1].strip() == "qc-prompt-align-hold"]
                if len(jobs) > 1:
                    raise RuntimeError("More than one aligned-question holdout job exists")
                if jobs:
                    job = jobs[0]
                else:
                    sent = subprocess.run(["sbatch", "--parsable", "--comment=aligned-question-sd-holdout",
                        str(root / "job.sbatch")], check=True, capture_output=True, text=True, timeout=30)
                    job = sent.stdout.strip().split(";", 1)[0]
                if not job.isdigit():
                    raise ValueError("Unexpected scheduler response")
                atomic(target, json.dumps(dict(job_id=job, gpus=1, models=manifest["models"],
                    cases_per_model=800, online_api_calls=0, created=time.time()), indent=2) + "\n")
    enable_holdout_timer(workspace, root)
    return dict(state="submitted", job_id=job, models=manifest["models"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--timer", required=True)
    args = parser.parse_args()
    try:
        result = promote(args.workspace, args.python)
    except Exception as exc:
        atomic(args.control_root / "monitor_status.json", json.dumps(dict(stage="monitor_error",
            error_type=type(exc).__name__, updated=time.time()), indent=2) + "\n")
        raise
    atomic(args.control_root / "monitor_status.json", json.dumps(dict(result, updated=time.time()), indent=2) + "\n")
    print(json.dumps(result, default=str))
    if result["state"] in ("not_eligible", "submitted", "prerequisite_terminal"):
        subprocess.run(["systemctl", "--user", "stop", args.timer], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
