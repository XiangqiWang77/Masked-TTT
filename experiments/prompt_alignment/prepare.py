#!/usr/bin/env python3
"""Freeze and submit one matched 200-case-per-model aligned-question study."""
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

CPU_TESTS = ["tests/test_prompt_alignment.py", "tests/test_prompt_alignment_runner.py",
             "tests/test_prompt_alignment_report.py", "tests/test_cli.py",
             "tests/test_batched_sd.py", "tests/test_memory.py",
             "tests/test_materialization_equivalence.py", "tests/test_kv_only_cartridge.py",
             "tests/test_gpt_request_date.py"]
NAME = "query_cartridges_prompt_aligned_dev_20261104"


def validate_protocol(source, profile):
    development, holdout = source["development_case_ids"], source["holdout_case_ids"]
    validate_partition(development, holdout)
    if (set(source["case_ids"]) != set(development) or len(development) != 200 or
        source["models"] != ["qwen", "gpt"] or source["keep_fraction"] != .1 or
        source["selection_policy"] != "decision_entity_link_v1" or
        source["entity_reserve_fraction"] != .25 or source["student_rollout_tokens"] != 32 or
        source["selected_transfer_weight"] != 0 or source["reference_answers_used_in_inference"] or
        profile["steps"] != 2 or profile["student_rollout_tokens"] != 8 or
        profile["gain_learning_rate"] != .01 or profile["value_learning_rate"] != .0001 or
        profile["self_question_count"] != 2):
        raise ValueError("The OLMo-linked two-step baseline changed")
    return dict(source, scope="Same-question-format on-policy SD, matched 200-case development",
                step_budgets=[2], align_student_teacher_prompt=True,
                standalone_answer_format="original", baseline_checkpoint=2,
                cached_scores=True, online_api_calls=0, max_gpu_hours=36)


def batch_script(root):
    script = slurm_script(root)
    replacements = {"qc-gqa-gain-dev": "qc-prompt-align-dev",
                    "gqa_head_gain/execute.py": "prompt_alignment/execute.py"}
    for old, new in replacements.items():
        if script.count(old) != 1:
            raise ValueError("Unexpected one-GPU script")
        script = script.replace(old, new, 1)
    return isolate_gpu_preflight(script)


def report_stub(workspace, root):
    refs = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    arguments = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
        "--baseline-root", str(workspace / "query_cartridges_longsteps_200dev_20261104"),
        "--references", str(refs), "--scorer-dir", str(root / "runtime/scorer"), "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport fcntl\nfrom pathlib import Path\nimport sys\n"
            f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
            "from experiments.prompt_alignment.report import main\n"
            f"sys.argv = [sys.argv[0], *{arguments!r}]\n"
            f"with Path({str(root / 'report.lock')!r}).open('a+') as lock:\n"
            "    fcntl.flock(lock, fcntl.LOCK_EX)\n    main()\n")


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != NAME:
        raise ValueError("Use the dedicated aligned-question development directory")
    source_root = workspace / "query_cartridges_longsteps_200dev_20261104"
    source = json.loads((source_root / "manifest.json").read_text())
    profile_root = workspace / "query_cartridges_value_lr_20261104"
    profile = json.loads((profile_root / "olmo_profile.json").read_text())
    manifest = validate_protocol(source, profile)
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("Do not modify a submitted or evaluated experiment")
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
            driver = workspace / "query_cartridges_rollout32_olmo_dev_20261104/run.py"
            text = driver.read_text()
            if text.count("manifest.get('student_rollout_tokens')!=32") != 1 or "MASKED_TTT_FIXED_CHAT_DATE" not in text:
                raise ValueError("Unexpected frozen 32-token driver")
            old_probe = "precision=CROSS/'query_cartridges_precision_probe_20261104'"
            if text.count(old_probe) != 1:
                raise ValueError("Unexpected optional precision probe location")
            text = text.replace(old_probe, "precision=ROOT/'disabled_optional_probe'", 1)
            atomic(runtime / "rollout_driver.py", text)
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
            raise RuntimeError("Frozen CPU validation failed:\n" + check.stdout + check.stderr)
        data = dict(cpu_tests_passed=True, passed_count=int(matched.group(1)), tests=CPU_TESTS,
                    gpu_preflight_required=True, submitted=False, online_api_calls=0)
        atomic(root / "cpu_validation_passed.json", json.dumps(data, indent=2) + "\n")
        return data


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != NAME:
        raise ValueError("Unexpected aligned-question job directory")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        if not json.loads((root / "cpu_validation_passed.json").read_text())["cpu_tests_passed"]:
            raise ValueError("Frozen CPU tests have not passed")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU cap is full")
            result = subprocess.run(["sbatch", "--parsable", "--comment=aligned-question-sd",
                                     str(root / "job.sbatch")], check=True, capture_output=True,
                                    text=True, timeout=30)
            job = result.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, models=["qwen", "gpt"], cases_per_model=200,
                        cached_scores=True, online_api_calls=0, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def enable_monitor(workspace, root):
    unit = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    timer = "query-cartridges-prompt-aligned-report.timer"
    service = ("[Unit]\nDescription=Report matched aligned-question SD development\n\n"
        "[Service]\nType=oneshot\n"
        "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
        f"ExecStart=/usr/bin/python3 {root}/runtime/repo/experiments/prompt_alignment/monitor.py "
        f"--workspace {workspace} --job-root {root} --timer {timer}\n")
    timer_text = ("[Unit]\nDescription=Update matched aligned-question SD once per hour\n\n"
        "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
        "[Install]\nWantedBy=timers.target\n")
    atomic(unit / "query-cartridges-prompt-aligned-report.service", service)
    atomic(unit / timer, timer_text)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", timer], check=True, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "job-root", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.job_root, args.python), indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace, args.job_root), indent=2))
        enable_monitor(args.workspace.resolve(), args.job_root.resolve())


if __name__ == "__main__":
    main()
