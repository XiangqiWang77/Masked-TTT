#!/usr/bin/env python3
"""Freeze and submit one GPT-OSS original-SD Value-rate development job."""
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
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.prepare import atomic, slurm_script
from experiments.long_rollout_steps.extend import isolate_gpu_preflight
from experiments.value_lr_physical.prepare import physical_driver

NAME = "query_cartridges_strong_value_gpt_dev_20261104"
CPU_TESTS = ["tests/test_strong_value_dev_runner.py", "tests/test_strong_value_dev_report.py",
             "tests/test_strong_value_dev_prepare.py", "tests/test_strong_value_dev_monitor.py",
             "tests/test_cli.py", "tests/test_frozen_scores.py",
             "tests/test_batched_sd.py", "tests/test_memory.py", "tests/test_hybrid_backbones.py",
             "tests/test_materialization_equivalence.py", "tests/test_kv_only_cartridge.py",
             "tests/test_gpt_request_date.py"]


def make_manifest(source, profile):
    development, holdout = source["case_ids"], source["holdout_case_ids"]
    validate_partition(development, holdout)
    if (len(development) != 200 or len(set(development)) != 200 or source["keep_fraction"] != .1 or
        source["selection_policy"] != "cached_decision_score10" or source["student_rollout_tokens"] != 32 or
        source["selected_transfer_weight"] != 0 or source["reference_answers_used_in_inference"] is not False or
        source["online_api_calls"] != 0 or profile["steps"] != 2 or
        profile["student_rollout_tokens"] != 8 or profile["value_learning_rate"] != .0001 or
        profile["gain_learning_rate"] != .01 or profile["self_question_count"] != 2):
        raise ValueError("The fixed original-SD source protocol changed")
    return dict(source, scope="GPT-OSS original-SD 0.003 Value rate, 200-case development",
        mode="development", case_count=200, models=["gpt"], case_ids=list(development),
        development_case_ids=list(development), holdout_case_ids=list(holdout), student_rollout_tokens=8,
        fixed_value_learning_rate=.003, gain_head_mode="kv", steps=2,
        holdout_evaluated_at_selection=False, max_gpu_hours=36, online_api_calls=0)


def batch_script(root):
    script = slurm_script(root)
    changes = {
        "qc-gqa-gain-dev": "qc-strong-value-dev",
        "gqa_head_gain/execute.py": "value_lr_physical/strong_dev_execute.py",
        "#SBATCH --partition=gpu,gpu_h100,gpu_h200,gpu_b200,gpu_rtx6000,scavenge_gpu":
            "#SBATCH --partition=gpu_h200,gpu_h100,gpu_b200,gpu_rtx6000,scavenge_gpu",
        "#SBATCH --constraint=gpu:a40|gpu:l40s|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell":
            "#SBATCH --constraint=gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell",
        "#SBATCH --time-min=00:30:00": "#SBATCH --time-min=00:20:00",
    }
    for old, new in changes.items():
        if script.count(old) != 1:
            raise ValueError("Unexpected one-GPU development script")
        script = script.replace(old, new, 1)
    return isolate_gpu_preflight(script)


def report_stub(workspace, root):
    references = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    args = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
            "--baseline-root", str(workspace / "query_cartridges_value_lr_20261104"),
            "--references", str(references), "--scorer-dir", str(root / "runtime/scorer"),
            "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport fcntl\nfrom pathlib import Path\nimport sys\n"
            f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
            "from experiments.value_lr_physical.strong_dev_report import main\n"
            f"sys.argv = [sys.argv[0], *{args!r}]\n"
            f"with Path({str(root / 'report.lock')!r}).open('a+') as lock:\n"
            "    fcntl.flock(lock, fcntl.LOCK_EX)\n    main()\n")


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != NAME:
        raise ValueError("Use the dedicated stronger-Value development directory")
    source_root = workspace / "query_cartridges_rollout32_olmo_dev_20261104"
    source = json.loads((source_root / "manifest.json").read_text())
    profile_root = workspace / "query_cartridges_value_lr_20261104"
    profile = json.loads((profile_root / "olmo_profile.json").read_text())
    manifest = make_manifest(source, profile)
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("Do not modify a submitted or evaluated development job")
        runtime = root / "runtime"
        if runtime.exists():
            if not (root / "manifest.json").is_file() or json.loads((root / "manifest.json").read_text()) != manifest:
                raise ValueError("The existing frozen runtime has a different protocol")
        else:
            frozen = runtime / "repo"
            frozen.mkdir(parents=True)
            ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
            for folder in ("src", "experiments", "tests", "examples"):
                shutil.copytree(REPO / folder, frozen / folder, ignore=ignore)
            atomic(runtime / "rollout_driver.py", physical_driver((source_root / "run.py").read_text()))
            shutil.copy2(profile_root / "olmo_profile.json", runtime / "olmo_profile.json")
            shared = workspace / "query_cartridges_linked_roll32_dev_20261104/runtime"
            shutil.copytree(shared / "helpers", runtime / "helpers", ignore=ignore)
            shutil.copytree(shared / "scorer", runtime / "scorer", ignore=ignore)
            (root / "logs").mkdir(exist_ok=True)
            atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
            atomic(root / "report.py", report_stub(workspace, root))
            atomic(root / "job.sbatch", batch_script(root))
        frozen = runtime / "repo"
        env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        check = subprocess.run([str(python), "-m", "pytest", "--rootdir", str(frozen),
            "--confcutdir", str(frozen), "-q", *CPU_TESTS], cwd=frozen, env=env,
            text=True, capture_output=True, timeout=180)
        matched = re.search(r"\b(\d+) passed\b", check.stdout)
        if check.returncode or not matched:
            raise RuntimeError("Frozen development validation failed:\n" + check.stdout + check.stderr)
        data = dict(cpu_tests_passed=True, passed_count=int(matched.group(1)), tests=CPU_TESTS,
                    gpu_preflight_required=True, online_api_calls=0, created=time.time())
        atomic(root / "cpu_validation_passed.json", json.dumps(data, indent=2) + "\n")
        return data


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != NAME:
        raise ValueError("Unexpected stronger-Value development directory")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        if not json.loads((root / "cpu_validation_passed.json").read_text())["cpu_tests_passed"]:
            raise ValueError("Frozen CPU validation has not passed")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU cap is full")
            sent = subprocess.run(["sbatch", "--parsable", "--comment=original-sd-strong-value-dev",
                                   str(root / "job.sbatch")], check=True, capture_output=True, text=True, timeout=30)
            job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, model="gpt", cases=200, value_learning_rate=.003,
                        cached_scores=True, online_api_calls=0, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def enable_monitor(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != NAME:
        raise ValueError("Unexpected stronger-Value development directory")
    unit = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    timer_name = "query-cartridges-strong-value-dev.timer"
    service = ("[Unit]\nDescription=Report original-SD stronger-Value GPT-OSS development\n\n"
        "[Service]\nType=oneshot\n"
        "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
        f"ExecStart=/usr/bin/python3 {root}/runtime/repo/experiments/value_lr_physical/strong_dev_monitor.py "
        f"--workspace {workspace} --job-root {root} --python {python} --timer {timer_name}\n")
    timer = ("[Unit]\nDescription=Update stronger-Value development once per hour\n\n"
        "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
        "[Install]\nWantedBy=timers.target\n")
    atomic(unit / "query-cartridges-strong-value-dev.service", service)
    atomic(unit / timer_name, timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", timer_name],
                   check=True, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "job-root", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.job_root, args.python), indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace, args.job_root), indent=2))
        enable_monitor(args.workspace, args.job_root, args.python)


if __name__ == "__main__":
    main()
