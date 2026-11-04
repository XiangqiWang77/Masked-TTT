#!/usr/bin/env python3
"""Freeze a stratified one-GPU pilot of longer SD and 2/4/8 update steps."""
import argparse
from collections import defaultdict
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

CPU_TESTS = ["tests/test_long_rollout_steps_prepare.py", "tests/test_long_rollout_steps.py", "tests/test_gpt_request_date.py",
             "tests/test_linked_long_rollout_physical.py", "tests/test_memory.py",
             "tests/test_batched_sd.py", "tests/test_kv_only_cartridge.py"]


def select_pilot(development):
    """All five QA3 sources and the first source of each other task, all lengths."""
    if len(development) != 200 or len(set(development)) != 200:
        raise ValueError("Expected 200 unique development cases")
    sources = defaultdict(set)
    source_cases = defaultdict(list)
    for case in development:
        parts = case.split("-", 2)
        if len(parts) != 3 or parts[0] != "babilong":
            raise ValueError("Unexpected case ID")
        source = parts[2]
        task = source.split("-", 1)[0]
        sources[task].add(source)
        source_cases[source].append(case)
    expected = {"qa" + str(index) for index in range(1, 11)}
    if set(sources) != expected or any(len(sources[task]) != 5 for task in expected):
        raise ValueError("Expected five development sources from each of ten QA types")
    for cases in source_cases.values():
        if {case.split("-", 2)[1] for case in cases} != {"8", "16", "32", "64"}:
            raise ValueError("Every development source needs all four context lengths")
    selected = set(sources["qa3"])
    selected.update(sorted(sources[task])[0] for task in expected if task != "qa3")
    result = sorted(case for source in selected for case in source_cases[source])
    if len(selected) != 14 or len(result) != 56 or len(set(result)) != 56:
        raise ValueError("Each of the fourteen selected sources must have four lengths")
    for source in selected:
        if {case.split("-", 2)[1] for case in source_cases[source]} != {"8", "16", "32", "64"}:
            raise ValueError("Every selected source needs all four context lengths")
    return result


def make_driver(source):
    old = "count not in (200,800)"
    if source.count(old) != 1 or "MASKED_TTT_FIXED_CHAT_DATE" not in source:
        raise ValueError("Unexpected rollout driver or missing request-date fix")
    return source.replace(old, "count not in (56,200,800)", 1)


def pilot_slurm(root):
    script = slurm_script(root)
    replacements = {"qc-gqa-gain-dev": "qc-longsteps-dev",
                    "gqa_head_gain/execute.py": "long_rollout_steps/execute.py"}
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
            "from experiments.long_rollout_steps.report import main\n"
            f"sys.argv = [sys.argv[0], *{args!r}]\nmain()\n")


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_longsteps_pilot_20261104":
        raise ValueError("Use the dedicated pilot directory")
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("Do not change a submitted or evaluated pilot")
        linked = json.loads((workspace / "query_cartridges_linked_selection_20261104/manifest.json").read_text())
        original_root = workspace / "query_cartridges_rollout32_olmo_dev_20261104"
        original = json.loads((original_root / "manifest.json").read_text())
        profile_root = workspace / "query_cartridges_value_lr_20261104"
        profile = json.loads((profile_root / "olmo_profile.json").read_text())
        development, holdout = linked["case_ids"], linked["holdout_case_ids"]
        validate_partition(development, holdout)
        if (development != original["case_ids"] or holdout != original["holdout_case_ids"] or
            linked["model_protocols"] != original["model_protocols"] or linked["keep_fraction"] != .1 or
            linked["selection_policy"] != "decision_entity_link_v1" or linked["entity_reserve_fraction"] != .25 or
            linked["selected_transfer_weight"] != 0 or profile["student_rollout_tokens"] != 8 or
            profile["steps"] != 2 or profile["gain_learning_rate"] != .01 or
            profile["value_learning_rate"] != .0001 or profile["self_question_count"] != 2):
            raise ValueError("The OLMo reference or linked selector settings changed")
        cases = select_pilot(development)
        manifest = dict(original, mode="development", scope="Long on-policy SD with matched step checkpoints",
            case_count=56, case_ids=cases, development_case_ids=list(development), holdout_case_ids=list(holdout),
            models=["qwen", "gpt"], selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25,
            student_rollout_tokens=32, step_budgets=[2, 4, 8], gain_head_mode="kv", max_gpu_hours=36,
            online_api_calls=0, case_selection_uses_answer_labels=False)
        runtime = root / "runtime"
        if runtime.exists():
            shutil.rmtree(runtime)
        runtime.mkdir()
        frozen = runtime / "repo"
        frozen.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
        for folder in ("src", "experiments", "tests"):
            shutil.copytree(REPO / folder, frozen / folder, ignore=ignore)
        atomic(runtime / "rollout_driver.py", make_driver((original_root / "run.py").read_text()))
        shutil.copy2(profile_root / "olmo_profile.json", runtime / "olmo_profile.json")
        shared = workspace / "query_cartridges_linked_roll32_dev_20261104/runtime"
        shutil.copytree(shared / "helpers", runtime / "helpers", ignore=ignore)
        shutil.copytree(shared / "scorer", runtime / "scorer", ignore=ignore)
        (root / "logs").mkdir(exist_ok=True)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "report.py", report_stub(workspace, root))
        atomic(root / "job.sbatch", pilot_slurm(root))
        env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        result = subprocess.run([str(python), "-m", "pytest", "-q", *CPU_TESTS], cwd=frozen,
                                env=env, text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError("Frozen pilot CPU validation failed:\n" + result.stdout + result.stderr)
        matched = re.search(r"\b(\d+) passed\b", result.stdout)
        if not matched:
            raise RuntimeError("No successful test count")
        data = dict(cpu_tests_passed=True, passed_count=int(matched.group(1)), tests=CPU_TESTS,
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
        check = json.loads((root / "cpu_validation_passed.json").read_text())
        if not check.get("cpu_tests_passed"):
            raise ValueError("CPU validation has not passed")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU cap is full")
            sent = subprocess.run(["sbatch", "--parsable", "--comment=long-sd-step-pilot",
                                   str(root / "job.sbatch")], check=True, capture_output=True,
                                  text=True, timeout=30)
            job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, models=["qwen", "gpt"], cases_per_model=56,
                        student_rollout_tokens=32, step_budgets=[2, 4, 8], gain_head_mode="kv",
                        cached_scores=True, online_api_calls=0, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def enable_monitor(root):
    unit = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    service = ("[Unit]\nDescription=Report original-SD long-rollout step pilot\n\n"
               "[Service]\nType=oneshot\n"
               "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
               f"ExecStart=/usr/bin/python3 {root}/runtime/repo/experiments/long_rollout_steps/monitor.py --root {root}\n")
    timer = ("[Unit]\nDescription=Update the long-rollout step pilot once per hour\n\n"
             "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
             "[Install]\nWantedBy=timers.target\n")
    atomic(unit / "query-cartridges-longsteps-report.service", service)
    atomic(unit / "query-cartridges-longsteps-report.timer", timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "query-cartridges-longsteps-report.timer"],
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
