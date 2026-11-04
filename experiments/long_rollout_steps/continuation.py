#!/usr/bin/env python3
"""Hourly continuation from a completed step pilot through source-disjoint evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.prepare import atomic, slurm_script
from experiments.long_rollout_steps import extend
from experiments.long_rollout_steps.holdout_report import combine, write
from experiments.long_rollout_steps.selection import development_choices


HOLDOUT_FILES = (
    "experiments/long_rollout_steps/probe.py", "experiments/long_rollout_steps/runner.py",
    "experiments/long_rollout_steps/report.py", "experiments/long_rollout_steps/extend.py",
    "experiments/long_rollout_steps/selection.py", "experiments/long_rollout_steps/holdout_runner.py",
    "experiments/long_rollout_steps/holdout_execute.py", "experiments/long_rollout_steps/holdout_report.py",
    "experiments/long_rollout_steps/continuation.py",
    "tests/test_long_rollout_steps.py", "tests/test_long_rollout_steps_prepare.py",
    "tests/test_long_rollout_steps_extend.py", "tests/test_long_rollout_step_selection.py",
    "tests/test_long_rollout_holdout_runner.py", "tests/test_long_rollout_holdout_report.py",
    "tests/test_long_rollout_extension_integration.py", "tests/test_long_rollout_continuation.py",
)
CPU_TESTS = ("tests/test_long_rollout_steps.py", "tests/test_long_rollout_steps_prepare.py",
    "tests/test_long_rollout_steps_extend.py", "tests/test_long_rollout_step_selection.py",
    "tests/test_long_rollout_holdout_runner.py", "tests/test_long_rollout_holdout_report.py",
    "tests/test_long_rollout_extension_integration.py", "tests/test_long_rollout_continuation.py",
    "tests/test_gpt_request_date.py")


def make_holdout_manifest(development, report):
    choices = development_choices(development, report)
    models = [model for model in development["models"] if choices[model]["eligible"]]
    if not models:
        return None, choices
    validate_partition(development["development_case_ids"], development["holdout_case_ids"])
    manifest = dict(development, mode="holdout", scope="Fixed-step original-SD source-disjoint holdout",
        case_count=800, case_ids=list(development["holdout_case_ids"]), models=models,
        fixed_final_base_steps={model: choices[model]["fixed_final_base_step"] for model in models},
        development_choices=choices, development_results_seen=True,
        holdout_evaluated_at_selection=False)
    return manifest, choices


def holdout_report_stub(workspace, root):
    references = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    args = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
            "--references", str(references), "--scorer-dir", str(root / "runtime/scorer"),
            "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport sys\n"
            f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
            "from experiments.long_rollout_steps.holdout_report import main\n"
            f"sys.argv = [sys.argv[0], *{args!r}]\nmain()\n")


def holdout_slurm(root):
    script = slurm_script(root)
    replacements = {"qc-gqa-gain-hold": "qc-longsteps-hold",
                    "gqa_head_gain/execute.py": "long_rollout_steps/holdout_execute.py"}
    for old, new in replacements.items():
        if script.count(old) != 1:
            raise ValueError("Unexpected single-GPU holdout script")
        script = script.replace(old, new, 1)
    return extend.isolate_gpu_preflight(script)


def prepare_holdout(workspace, dev_root, root, python, manifest):
    workspace, dev_root, root = workspace.resolve(), dev_root.resolve(), root.resolve()
    if (dev_root.parent != workspace or dev_root.name != "query_cartridges_longsteps_200dev_20261104" or
        root.parent != workspace or root.name != "query_cartridges_longsteps_holdout_20261104"):
        raise ValueError("Unexpected development or holdout directory")
    if not (dev_root / "finished.json").exists():
        raise RuntimeError("The full checkpoint development has not completed")
    actual, _ = make_holdout_manifest(json.loads((dev_root / "manifest.json").read_text()),
                                       json.loads((dev_root / "latest.json").read_text()))
    if actual != manifest or manifest is None:
        raise ValueError("Holdout configuration differs from development-selected steps")
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "manifest.json").exists():
            if json.loads((root / "manifest.json").read_text()) != manifest:
                raise ValueError("The existing holdout has a different fixed configuration")
        if (root / "submission.json").exists():
            if not (root / "cpu_validation_passed.json").exists():
                raise ValueError("A submitted holdout has no validation record")
            return root
        runtime = root / "runtime"
        if not runtime.exists():
            stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
            try:
                staged = stage / "runtime"
                shutil.copytree(dev_root / "runtime", staged,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
                for relative in HOLDOUT_FILES:
                    target = staged / "repo" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(REPO / relative, target)
                atomic(staged / "holdout_provenance.json", json.dumps(
                    dict(source=str(dev_root), steps=manifest["fixed_final_base_steps"],
                         on_policy_sd_unchanged=True, online_api_calls=0), indent=2) + "\n")
                os.replace(staged, runtime)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        elif not (runtime / "holdout_provenance.json").exists():
            raise ValueError("The existing holdout runtime lacks provenance")
        (root / "logs").mkdir(exist_ok=True)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "report.py", holdout_report_stub(workspace, root))
        atomic(root / "job.sbatch", holdout_slurm(root))
        env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        test_root = runtime / "repo"
        result = subprocess.run([str(python), "-m", "pytest", "--rootdir", str(test_root),
                                 "--confcutdir", str(test_root), "-q", *CPU_TESTS],
            cwd=runtime / "repo", env=env, text=True, capture_output=True, timeout=180)
        matched = re.search(r"\b(\d+) passed\b", result.stdout)
        if result.returncode or not matched:
            raise RuntimeError("Frozen holdout validation failed:\n" + result.stdout + result.stderr)
        atomic(root / "cpu_validation_passed.json", json.dumps(
            dict(passed=True, passed_count=int(matched.group(1)), tests=CPU_TESTS,
                 gpu_preflight_required=True, steps=manifest["fixed_final_base_steps"],
                 online_api_calls=0), indent=2) + "\n")
    return root


def submit_holdout(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_longsteps_holdout_20261104":
        raise ValueError("Unexpected source-disjoint holdout directory")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        check = json.loads((root / "cpu_validation_passed.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if not check["passed"] or check["steps"] != manifest["fixed_final_base_steps"]:
            raise ValueError("The frozen holdout did not pass validation")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU job cap is full; retry on the next interval")
            sent = subprocess.run(["sbatch", "--parsable", "--job-name=qc-longsteps-holdout",
                "--comment=long-sd-fixed-step-holdout", str(root / "job.sbatch")],
                check=True, capture_output=True, text=True, timeout=30)
            job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, models=manifest["models"], cases_per_model=800,
                        fixed_final_base_steps=manifest["fixed_final_base_steps"],
                        cached_scores=True, online_api_calls=0, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def _refresh(root):
    subprocess.run([sys.executable, str(root / "report.py")], check=True,
                   timeout=120, capture_output=True, text=True)
    return json.loads((root / "latest.json").read_text())


def _pilot_snapshot(workspace, pilot):
    """Report to a private stem so the pilot's own hourly timer cannot race us."""
    stem = workspace / "query_cartridges_longsteps_continuation_pilot_snapshot"
    repo = pilot / "runtime/repo"
    references = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    command = [sys.executable, str(repo / "experiments/long_rollout_steps/report.py"),
        "--manifest", str(pilot / "manifest.json"), "--candidate-root", str(pilot),
        "--references", str(references), "--scorer-dir", str(pilot / "runtime/scorer"),
        "--output", str(stem)]
    env = dict(os.environ, PYTHONPATH=str(repo))
    subprocess.run(command, check=True, timeout=120, capture_output=True, text=True, env=env)
    return json.loads(stem.with_suffix(".json").read_text())


def advance(workspace, python):
    """One interval: reuse the pilot, then select and evaluate fixed steps."""
    workspace = workspace.resolve()
    pilot = workspace / "query_cartridges_longsteps_pilot_20261104"
    dev = workspace / "query_cartridges_longsteps_200dev_20261104"
    hold = workspace / "query_cartridges_longsteps_holdout_20261104"
    if not (pilot / "finished.json").exists():
        return dict(stage="pilot_in_progress")
    pilot_report = _pilot_snapshot(workspace, pilot)
    extend.prepare(workspace, pilot, dev, pilot_report=pilot_report)
    submitted_dev = extend.submit(workspace, dev)
    development = _refresh(dev)
    if not (dev / "finished.json").exists():
        return dict(stage="development_in_progress", job_id=submitted_dev["job_id"])
    dev_manifest = json.loads((dev / "manifest.json").read_text())
    manifest, choices = make_holdout_manifest(dev_manifest, development)
    atomic(dev / "development_choices.json", json.dumps(choices, indent=2) + "\n")
    if manifest is None:
        return dict(stage="no_model_passed_development", choices=choices)
    prepare_holdout(workspace, dev, hold, python, manifest)
    submitted_hold = submit_holdout(workspace, hold)
    holdout = _refresh(hold)
    if not (hold / "finished.json").exists():
        return dict(stage="holdout_in_progress", job_id=submitted_hold["job_id"],
                    fixed_final_base_steps=manifest["fixed_final_base_steps"])
    combined = combine(dev_manifest, development, manifest, holdout)
    if combined is None:
        raise ValueError("Finished holdout lacks complete scored rows")
    write(combined, dev / "combined_1000")
    return dict(stage="complete", models=manifest["models"],
                report=str(dev / "combined_1000.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--timer", default="query-cartridges-longsteps-continuation.timer")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    with (workspace / "query_cartridges_longsteps_continuation.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            result = advance(workspace, args.python)
        except Exception as exc:
            atomic(workspace / "query_cartridges_longsteps_continuation_status.json",
                   json.dumps(dict(stage="monitor_error", error_type=type(exc).__name__,
                                   updated=time.time()), indent=2) + "\n")
            raise
        atomic(workspace / "query_cartridges_longsteps_continuation_status.json",
               json.dumps(dict(result, updated=time.time()), indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if result["stage"] in ("complete", "no_model_passed_development"):
            subprocess.run(["systemctl", "--user", "stop", args.timer],
                           check=True, capture_output=True, text=True)


if __name__ == "__main__":
    main()
