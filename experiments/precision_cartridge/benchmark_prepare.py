#!/usr/bin/env python3
"""Freeze a one-GPU same-training-state BF16/FP32 materialization pilot."""
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
from experiments.long_rollout_steps.prepare import pilot_slurm, select_pilot

CPU_TESTS = ["tests/test_precision_benchmark_prepare.py", "tests/test_precision_benchmark.py", "tests/test_paired_precision_diagnostic.py",
             "tests/test_precision_cartridge.py", "tests/test_gpt_request_date.py",
             "tests/test_linked_long_rollout_physical.py", "tests/test_memory.py",
             "tests/test_batched_sd.py", "tests/test_kv_only_cartridge.py"]
CPU_BATCHES = [CPU_TESTS[:4], CPU_TESTS[4:7], CPU_TESTS[7:]]


def precision_slurm(root):
    script = pilot_slurm(root)
    replacements = {"qc-longsteps-dev": "qc-precision-dev",
                    "long_rollout_steps/execute.py": "precision_cartridge/benchmark_execute.py"}
    for old, new in replacements.items():
        if script.count(old) != 1:
            raise ValueError("Unexpected shared Slurm script")
        script = script.replace(old, new, 1)
    return script


def report_stub(workspace, root):
    references = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    args = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
            "--references", str(references), "--scorer-dir", str(root / "runtime/scorer"),
            "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport sys\n"
            f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
            "from experiments.precision_cartridge.benchmark_report import main\n"
            f"sys.argv = [sys.argv[0], *{args!r}]\nmain()\n")


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_precision_pilot_20261104":
        raise ValueError("Use the dedicated precision pilot directory")
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("Do not change a submitted or evaluated pilot")
        earlier = workspace / "query_cartridges_longsteps_pilot_20261104"
        original = json.loads((earlier / "manifest.json").read_text())
        if (original["case_count"] != 56 or original["case_ids"] != select_pilot(original["development_case_ids"]) or
            original["selection_policy"] != "decision_entity_link_v1" or original["entity_reserve_fraction"] != .25 or
            original["student_rollout_tokens"] != 32 or original["gain_head_mode"] != "kv" or
            original["selected_transfer_weight"] != 0 or original["online_api_calls"] != 0):
            raise ValueError("The matched long-rollout pilot protocol changed")
        manifest = dict(original, scope="Same-state BF16 versus FP32 cartridge materialization",
            precision_comparison="same_trained_state_bf16_fp32", base_steps=2,
            case_selection_uses_answer_labels=False)
        manifest.pop("step_budgets", None)
        runtime = root / "runtime"
        if runtime.exists():
            shutil.rmtree(runtime)
        runtime.mkdir()
        frozen = runtime / "repo"
        frozen.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
        for folder in ("src", "experiments", "tests"):
            shutil.copytree(REPO / folder, frozen / folder, ignore=ignore)
        old_runtime = earlier / "runtime"
        driver = (old_runtime / "rollout_driver.py").read_text()
        if "count not in (56,200,800)" not in driver or "MASKED_TTT_FIXED_CHAT_DATE" not in driver:
            raise ValueError("Unexpected frozen pilot driver")
        atomic(runtime / "rollout_driver.py", driver)
        shutil.copy2(old_runtime / "olmo_profile.json", runtime / "olmo_profile.json")
        shutil.copytree(old_runtime / "helpers", runtime / "helpers", ignore=ignore)
        shutil.copytree(old_runtime / "scorer", runtime / "scorer", ignore=ignore)
        (root / "logs").mkdir(exist_ok=True)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "report.py", report_stub(workspace, root))
        atomic(root / "job.sbatch", precision_slurm(root))
        env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        passed = 0
        # Running all tiny-model suites in one pytest process can keep model
        # graphs resident on the shared login node. Bound memory by executing
        # separate sequential test processes; every listed suite still runs.
        for batch in CPU_BATCHES:
            result = subprocess.run([str(python), "-m", "pytest", "-q", *batch], cwd=frozen,
                                    env=env, text=True, capture_output=True, timeout=180)
            if result.returncode:
                raise RuntimeError(f"Frozen precision CPU validation failed ({result.returncode}):\n" +
                                   result.stdout + result.stderr)
            matched = re.search(r"\b(\d+) passed\b", result.stdout)
            if not matched:
                raise RuntimeError("No successful test count")
            passed += int(matched.group(1))
        data = dict(cpu_tests_passed=True, passed_count=passed, tests=CPU_TESTS,
                    gpu_preflight_required=True, submitted=False, online_api_calls=0)
        atomic(root / "cpu_validation_passed.json", json.dumps(data, indent=2) + "\n")
        return data


def submit(workspace, root):
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        if not json.loads((root / "cpu_validation_passed.json").read_text()).get("cpu_tests_passed"):
            raise ValueError("CPU validation has not passed")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU cap is full")
            sent = subprocess.run(["sbatch", "--parsable", "--comment=precision-materialization-pilot",
                                   str(root / "job.sbatch")], check=True, capture_output=True,
                                  text=True, timeout=30)
            job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, models=["qwen", "gpt"], cases_per_model=56,
                        student_rollout_tokens=32, base_steps=2, gain_head_mode="kv",
                        same_training_state=True, cached_scores=True, online_api_calls=0, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def enable_monitor(root):
    unit = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    service = ("[Unit]\nDescription=Report same-state BF16/FP32 Cartridge pilot\n\n"
               "[Service]\nType=oneshot\n"
               "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
               f"ExecStart=/usr/bin/python3 {root}/runtime/repo/experiments/precision_cartridge/benchmark_monitor.py --root {root}\n")
    timer = ("[Unit]\nDescription=Update the precision pilot once per hour\n\n"
             "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
             "[Install]\nWantedBy=timers.target\n")
    atomic(unit / "query-cartridges-precision-report.service", service)
    atomic(unit / "query-cartridges-precision-report.timer", timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "query-cartridges-precision-report.timer"],
                   check=True, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.job_root, args.python), indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace.resolve(), args.job_root.resolve()), indent=2))
        enable_monitor(args.job_root.resolve())
        print("Hourly pilot reporting enabled")


if __name__ == "__main__":
    main()
