#!/usr/bin/env python3
"""Freeze, validate and submit the development-fixed physical 8-step holdout."""
import argparse
import fcntl
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
from experiments.gqa_head_gain.prepare import atomic
from experiments.step_physical.protocol import holdout_manifest
from experiments.value_lr_physical.prepare import physical_driver, physical_slurm

CPU_TESTS = ["tests/test_step_physical.py", "tests/test_step_physical_prepare.py", "tests/test_step_physical_monitor.py",
             "tests/test_value_lr_physical.py", "tests/test_value_lr_physical_report.py",
             "tests/test_value_lr_frozen_driver.py", "tests/test_value_lr_archive_equivalence.py",
             "tests/test_value_lr_recovery_report.py",
             "tests/test_long_rollout_steps.py", "tests/test_frozen_scores.py", "tests/test_cli.py",
             "tests/test_batched_sd.py", "tests/test_memory.py", "tests/test_hybrid_backbones.py",
             "tests/test_materialization_equivalence.py", "tests/test_kv_only_cartridge.py",
             "tests/test_gpt_request_date.py"]


def job_script(root):
    script = physical_slurm(root, "gpt")
    for old, new in (("qc-value-physical-gpt", "qc-physical-step-gpt"),
                     ("value_lr_physical/execute.py", "step_physical/execute.py")):
        if script.count(old) != 1:
            raise ValueError("The single-GPU physical job template changed")
        script = script.replace(old, new, 1)
    return script


def report_stub(workspace, root):
    references = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    args = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
            "--references", str(references), "--scorer-dir", str(root / "runtime/scorer"),
            "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport fcntl\nfrom pathlib import Path\nimport sys\n"
            f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
            "from experiments.step_physical.report import main\n"
            f"sys.argv = [sys.argv[0], *{args!r}]\n"
            f"with Path({str(root / 'report.lock')!r}).open('a+') as lock:\n"
            "    fcntl.flock(lock, fcntl.LOCK_EX)\n    main()\n")


def validate_frozen(root, python, manifest):
    runtime, frozen = root / "runtime", root / "runtime/repo"
    required = [root / "manifest.json", root / "development_manifest.json", root / "development_snapshot.json",
                root / "report.py", root / "job.sbatch", runtime / "rollout_driver.py",
                runtime / "olmo_profile.json", runtime / "helpers", runtime / "scorer", frozen / "src",
                frozen / "experiments", frozen / "tests", frozen / "examples"]
    if any(not path.exists() for path in required):
        raise ValueError("The unfinished frozen step runtime is incomplete")
    if json.loads((root / "manifest.json").read_text()) != manifest:
        raise ValueError("The unfinished frozen step runtime uses a different protocol")
    env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CARTRIDGE_TEST_BASE_STEPS="8",
               CARTRIDGE_VALUE_DRIVER_PATH=str(runtime / "rollout_driver.py"))
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_API_KEY_FILE", None)
    checked = subprocess.run([str(python), "-m", "pytest", "--rootdir", str(frozen),
        "--confcutdir", str(frozen), "-q", *CPU_TESTS], cwd=frozen, env=env,
        text=True, capture_output=True, timeout=240)
    matched = re.search(r"\b(\d+) passed\b", checked.stdout)
    if checked.returncode or not matched:
        raise RuntimeError("Frozen physical step validation failed:\n" + checked.stdout + checked.stderr)
    data = dict(cpu_tests_passed=True, passed_count=int(matched.group(1)), tests=CPU_TESTS,
                gpu_preflight_required=True, model="gpt", fixed_base_steps=8,
                fixed_value_learning_rate=.0001, online_api_calls=0, created=time.time())
    atomic(root / "cpu_validation_passed.json", json.dumps(data, indent=2) + "\n")
    return data


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_step_physical_gpt_holdout_20261104":
        raise ValueError("Use the dedicated GPT physical step holdout directory")
    dev = workspace / "query_cartridges_step_dev_20261104"
    if not (dev / "finished.json").exists():
        raise ValueError("The full step development must be complete")
    development = json.loads((dev / "manifest.json").read_text())
    report = json.loads((dev / "latest.json").read_text())
    profile_path = workspace / "query_cartridges_value_lr_20261104/olmo_profile.json"
    profile = json.loads(profile_path.read_text())
    manifest = holdout_manifest(development, report, profile, "gpt")
    if manifest is None or manifest["steps"] != 8:
        raise ValueError("The completed development did not select the 8-step GPT candidate")
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validation = root / "cpu_validation_passed.json"
        if validation.exists():
            if json.loads((root / "manifest.json").read_text()) != manifest:
                raise ValueError("Existing physical step protocol differs")
            data = json.loads(validation.read_text())
            if not data.get("cpu_tests_passed") or data.get("fixed_base_steps") != 8:
                raise ValueError("Existing physical step validation differs")
            return data
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("A submitted physical step holdout lacks validation")
        runtime = root / "runtime"
        if runtime.exists():
            return validate_frozen(root, python, manifest)
        runtime.mkdir()
        frozen = runtime / "repo"
        frozen.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
        for folder in ("src", "experiments", "tests", "examples"):
            shutil.copytree(REPO / folder, frozen / folder, ignore=ignore)
        driver = workspace / "query_cartridges_rollout32_olmo_dev_20261104/run.py"
        atomic(runtime / "rollout_driver.py", physical_driver(driver.read_text()))
        shutil.copy2(profile_path, runtime / "olmo_profile.json")
        helpers = workspace / "query_cartridges_linked_roll32_dev_20261104/runtime"
        shutil.copytree(helpers / "helpers", runtime / "helpers", ignore=ignore)
        shutil.copytree(helpers / "scorer", runtime / "scorer", ignore=ignore)
        (root / "logs").mkdir(exist_ok=True)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "development_manifest.json", json.dumps(development, indent=2) + "\n")
        atomic(root / "development_snapshot.json", json.dumps(report, indent=2) + "\n")
        atomic(root / "report.py", report_stub(workspace, root))
        atomic(root / "job.sbatch", job_script(root))
        return validate_frozen(root, python, manifest)


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_step_physical_gpt_holdout_20261104":
        raise ValueError("Unexpected physical step holdout directory")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        check = json.loads((root / "cpu_validation_passed.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if not check["cpu_tests_passed"] or check["fixed_base_steps"] != manifest["steps"]:
            raise ValueError("Frozen physical step tests are incomplete")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU job cap is full")
            sent = subprocess.run(["sbatch", "--parsable", "--comment=physical-original-sd-8-step",
                str(root / "job.sbatch")], check=True, capture_output=True, text=True, timeout=30)
            job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, model="gpt", cases=800, fixed_base_steps=8,
                        fixed_value_learning_rate=.0001, cached_scores=True, online_api_calls=0,
                        created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "job-root", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.job_root, args.python), indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace, args.job_root), indent=2))
        from experiments.step_physical.monitor import enable_monitor
        print("Hourly reporting:", enable_monitor(args.workspace.resolve(), args.job_root.resolve()))


if __name__ == "__main__":
    main()
