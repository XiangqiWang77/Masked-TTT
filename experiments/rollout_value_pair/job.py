#!/usr/bin/env python3
"""Freeze and submit one single-GPU same-model SD hyperparameter comparison."""
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
from experiments.self_study_prompt.study_gate import development_decision

DEV = "query_cartridges_rollout_value_qwen_dev_20261104"
HOLDOUT = "query_cartridges_rollout_value_qwen_holdout_20261104"
CPU_TESTS = ["tests/test_rollout_value_pair.py", "tests/test_rollout_value_job.py",
             "tests/test_same_gpu_pair.py", "tests/test_batched_sd.py", "tests/test_frozen_scores.py",
             "tests/test_materialization_equivalence.py", "tests/test_kv_only_cartridge.py"]


def make_manifest(rollout, linked, profile):
    development, holdout = linked["case_ids"], linked["holdout_case_ids"]
    validate_partition(development, holdout)
    if (development != rollout["case_ids"] or holdout != rollout["holdout_case_ids"] or
        rollout["model_protocols"] != linked["model_protocols"] or
        linked["selection_policy"] != "decision_entity_link_v1" or linked["entity_reserve_fraction"] != .25 or
        linked["keep_fraction"] != .1 or linked["reference_answers_used_in_inference"] is not False or
        linked["online_api_calls"] != 0 or profile["student_rollout_tokens"] != 8 or profile["steps"] != 2 or
        profile["gain_learning_rate"] != .01 or profile["value_learning_rate"] != .0001 or
        profile["self_question_count"] != 2 or profile["max_answer_steps"] != 8 or
        profile["answer_KL_stop"] != .5 or profile["guard_enabled"] is not True):
        raise ValueError("The fixed original-method baseline changed")
    return dict(rollout, scope="Qwen original-SD 8/.0001 versus 32/.001 on the same GPU",
        mode="development", case_count=200, case_ids=list(development), models=["qwen"],
        development_case_ids=list(development), holdout_case_ids=list(holdout),
        selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25,
        baseline_rollout_tokens=8, student_rollout_tokens=32,
        baseline_value_lr=.0001, candidate_value_lr=.001, selected_transfer_weight=0.,
        gain_head_mode="kv", steps=2, max_answer_steps=8, max_gpu_hours=36,
        holdout_evaluated_at_selection=False, online_api_calls=0,
        development_gate="Beat same-run whole, selected zero, ICL Teacher, and baseline cartridge",
        holdout_target="At least +3 percentage points versus ICL Teacher")


def batch_script(root):
    source = slurm_script(root)
    expected = "qc-gqa-gain-hold" if root.name == HOLDOUT else "qc-gqa-gain-dev"
    replacements = {expected: "qc-roll-value-hold" if root.name == HOLDOUT else "qc-roll-value-dev",
        "gqa_head_gain/execute.py": "rollout_value_pair/execute.py",
        "#SBATCH --time-min=00:30:00": "#SBATCH --time-min=00:20:00",
        "#SBATCH --mem=128G": "#SBATCH --mem=64G"}
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Unexpected single-GPU Slurm script")
        source = source.replace(old, new, 1)
    return isolate_gpu_preflight(source)


def report_stub(workspace, root):
    references = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    args = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
            "--references", str(references), "--scorer-dir", str(root / "runtime/scorer"),
            "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport fcntl\nfrom pathlib import Path\nimport sys\n"
        f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
        "from experiments.rollout_value_pair.report import main\n"
        f"sys.argv = [sys.argv[0], *{args!r}]\n"
        f"with Path({str(root / 'report.lock')!r}).open('a+') as lock:\n"
        "    fcntl.flock(lock, fcntl.LOCK_EX)\n    main()\n")


def validate_frozen(root, python):
    frozen = root / "runtime/repo"
    cache = root / "runtime_cache"
    (cache / "triton").mkdir(parents=True, exist_ok=True)
    (cache / "torch_extensions").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TRITON_CACHE_DIR=str(cache / "triton"), TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"))
    done = subprocess.run([str(python), "-m", "pytest", "--rootdir", str(frozen),
        "--confcutdir", str(frozen), "-q", *CPU_TESTS], cwd=frozen, env=env,
        text=True, capture_output=True, timeout=180)
    match = re.search(r"\b(\d+) passed\b", done.stdout)
    if done.returncode or not match:
        raise RuntimeError(f"Frozen rollout/Value pair tests exited {done.returncode}:\n" + done.stdout + done.stderr)
    result = dict(cpu_tests_passed=True, passed_count=int(match.group(1)), tests=CPU_TESTS,
                  gpu_preflight_required=True, online_api_calls=0, created=time.time())
    atomic(root / "cpu_validation_passed.json", json.dumps(result, indent=2) + "\n")
    return result


def prepare(workspace, root, python):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != DEV:
        raise ValueError("Use the dedicated rollout/Value development directory")
    rollout_root = workspace / "query_cartridges_rollout32_olmo_dev_20261104"
    linked_root = workspace / "query_cartridges_linked_selection_20261104"
    profile_root = workspace / "query_cartridges_value_lr_20261104"
    manifest = make_manifest(json.loads((rollout_root / "manifest.json").read_text()),
        json.loads((linked_root / "manifest.json").read_text()),
        json.loads((profile_root / "olmo_profile.json").read_text()))
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("Do not rewrite a submitted experiment")
        runtime = root / "runtime"
        if runtime.exists():
            if not (root / "manifest.json").is_file() or json.loads((root / "manifest.json").read_text()) != manifest:
                raise ValueError("An existing frozen protocol changed")
        else:
            frozen = runtime / "repo"
            frozen.mkdir(parents=True)
            ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
            for folder in ("src", "experiments", "tests", "examples"):
                shutil.copytree(REPO / folder, frozen / folder, ignore=ignore)
            source = (rollout_root / "run.py").read_text()
            optional = "precision=CROSS/'query_cartridges_precision_probe_20261104'"
            if source.count(optional) != 1 or "t['student_rollout_limit']!=32" not in source:
                raise ValueError("The 32-token source driver changed")
            atomic(runtime / "rollout_driver.py", source.replace(optional, "precision=ROOT/'disabled_optional_probe'", 1))
            shutil.copy2(profile_root / "olmo_profile.json", runtime / "olmo_profile.json")
            shared = workspace / "query_cartridges_linked_roll32_dev_20261104/runtime"
            shutil.copytree(shared / "helpers", runtime / "helpers", ignore=ignore)
            shutil.copytree(shared / "scorer", runtime / "scorer", ignore=ignore)
            (root / "logs").mkdir(exist_ok=True)
            atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
            atomic(root / "report.py", report_stub(workspace, root))
            atomic(root / "job.sbatch", batch_script(root))
        return validate_frozen(root, python)


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name not in (DEV, HOLDOUT):
        raise ValueError("Unexpected paired-hyperparameter directory")
    if root.name == HOLDOUT and not development_decision(
        json.loads((root / "development_snapshot.json").read_text()))["eligible"]:
        raise ValueError("The candidate did not beat its development controls")
    if not json.loads((root / "cpu_validation_passed.json").read_text())["cpu_tests_passed"]:
        raise ValueError("Frozen CPU tests did not pass")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU job cap is full")
            sent = subprocess.run(["sbatch", "--parsable", "--comment=rollout-value-paired",
                str(root / "job.sbatch")], check=True, capture_output=True, text=True, timeout=30)
            job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, model="qwen", cases=200 if root.name == DEV else 800,
                        cached_scores=True, online_api_calls=0, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def enable_monitor(workspace, root, python):
    unit = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    stage = "holdout" if root.name == HOLDOUT else "dev"
    timer = f"query-cartridges-rollout-value-{stage}.timer"
    service = ("[Unit]\nDescription=Report paired original-SD rollout and Value rate\n\n[Service]\nType=oneshot\n"
        "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
        f"ExecStart=/usr/bin/python3 {root}/runtime/repo/experiments/rollout_value_pair/monitor.py "
        f"--workspace {workspace} --job-root {root} --python {python} --timer {timer}\n")
    timer_text = ("[Unit]\nDescription=Update paired rollout/Value study hourly\n\n[Timer]\n"
        "OnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n[Install]\nWantedBy=timers.target\n")
    atomic(unit / f"query-cartridges-rollout-value-{stage}.service", service)
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
        enable_monitor(args.workspace.resolve(), args.job_root.resolve(), args.python.resolve())


if __name__ == "__main__":
    main()
