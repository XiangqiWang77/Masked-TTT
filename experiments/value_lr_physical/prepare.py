#!/usr/bin/env python3
"""Freeze and submit a one-model physical 800-case LR holdout."""
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
from experiments.gqa_head_gain.prepare import atomic, make_driver, slurm_script
from experiments.long_rollout_steps.extend import isolate_gpu_preflight
from experiments.value_lr_physical.protocol import holdout_manifest

CPU_TESTS = ["tests/test_value_lr_physical.py", "tests/test_value_lr_physical_report.py",
             "tests/test_value_lr_physical_prepare.py", "tests/test_value_lr_physical_monitor.py",
             "tests/test_value_lr_physical_handoff.py",
             "tests/test_value_lr_physical_launch.py",
             "tests/test_value_lr_frozen_driver.py",
             "tests/test_cli.py",
             "tests/test_batched_sd.py", "tests/test_memory.py",
             "tests/test_materialization_equivalence.py", "tests/test_kv_only_cartridge.py",
             "tests/test_gpt_request_date.py"]


def physical_driver(source):
    value = make_driver(source)
    old = "precision=CROSS/'query_cartridges_precision_probe_20261104'"
    if value.count(old) != 1:
        raise ValueError("Unexpected optional diagnostic location")
    return value.replace(old, "precision=ROOT/'disabled_optional_probe'", 1)


def physical_slurm(root, model):
    script = slurm_script(root)
    replacements = {"qc-gqa-gain-hold": "qc-value-physical-" + model,
                    "gqa_head_gain/execute.py": "value_lr_physical/execute.py"}
    for old, new in replacements.items():
        if script.count(old) != 1:
            raise ValueError("Unexpected one-GPU holdout script")
        script = script.replace(old, new, 1)
    if model == "gpt":
        old_partition = "#SBATCH --partition=gpu,gpu_h100,gpu_h200,gpu_b200,gpu_rtx6000,scavenge_gpu"
        new_partition = "#SBATCH --partition=gpu_rtx6000,gpu_b200,gpu_h200,gpu_h100,scavenge_gpu"
        if script.count(old_partition) != 1:
            raise ValueError("Unexpected GPU partition routing")
        script = script.replace(old_partition, new_partition, 1)
        old = "#SBATCH --constraint=gpu:a40|gpu:l40s|gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
        new = "#SBATCH --constraint=gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
        if script.count(old) != 1:
            raise ValueError("Unexpected GPT-OSS memory routing")
        script = script.replace(old, new, 1)
    return isolate_gpu_preflight(script)


def report_stub(workspace, root):
    references = workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json"
    args = ["--manifest", str(root / "manifest.json"), "--candidate-root", str(root),
            "--references", str(references), "--scorer-dir", str(root / "runtime/scorer"),
            "--output", str(root / "latest")]
    return ("#!/usr/bin/env python3\nimport fcntl\nfrom pathlib import Path\nimport sys\n"
            f"sys.path.insert(0, {str(root / 'runtime/repo')!r})\n"
            "from experiments.value_lr_physical.report import main\n"
            f"sys.argv = [sys.argv[0], *{args!r}]\n"
            f"with Path({str(root / 'report.lock')!r}).open('a+') as lock:\n"
            "    fcntl.flock(lock, fcntl.LOCK_EX)\n    main()\n")


def validate_frozen(root, python, model, manifest):
    """Validate an already-frozen runtime, including after a transient failure."""
    runtime, frozen = root / "runtime", root / "runtime/repo"
    required = [root / "manifest.json", root / "development_manifest.json",
                root / "development_snapshot.json", root / "report.py", root / "job.sbatch",
                runtime / "rollout_driver.py", runtime / "olmo_profile.json",
                runtime / "helpers", runtime / "scorer", frozen / "src", frozen / "experiments",
                frozen / "tests", frozen / "examples"]
    if any(not path.exists() for path in required):
        raise ValueError("The unfinished frozen runtime is incomplete")
    if json.loads((root / "manifest.json").read_text()) != manifest:
        raise ValueError("The unfinished frozen runtime uses a different protocol")
    env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CARTRIDGE_VALUE_DRIVER_PATH=str(runtime / "rollout_driver.py"))
    result = subprocess.run([str(python), "-m", "pytest", "--rootdir", str(frozen),
        "--confcutdir", str(frozen), "-q", *CPU_TESTS], cwd=frozen, env=env,
        text=True, capture_output=True, timeout=180)
    matched = re.search(r"\b(\d+) passed\b", result.stdout)
    if result.returncode or not matched:
        raise RuntimeError("Frozen physical holdout validation failed:\n" + result.stdout + result.stderr)
    data = dict(cpu_tests_passed=True, passed_count=int(matched.group(1)), tests=CPU_TESTS,
                gpu_preflight_required=True, model=model, fixed_value_learning_rate=manifest["fixed_value_learning_rate"],
                online_api_calls=0, created=time.time())
    atomic(root / "cpu_validation_passed.json", json.dumps(data, indent=2) + "\n")
    return data


def prepare(workspace, root, python, model):
    workspace, root = workspace.resolve(), root.resolve()
    expected_name = "query_cartridges_value_lr_physical_" + model + "_holdout_20261104"
    if model not in ("qwen", "gpt") or root.parent != workspace or root.name != expected_name:
        raise ValueError("Use the dedicated single-model physical holdout directory")
    development = workspace / "query_cartridges_value_lr_20261104"
    dev_manifest = json.loads((development / "manifest.json").read_text())
    dev_report = json.loads((development / "latest.json").read_text())
    profile = json.loads((development / "olmo_profile.json").read_text())
    manifest = holdout_manifest(dev_manifest, dev_report, profile, model)
    if manifest is None:
        raise ValueError("The completed model development did not beat the three controls")
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing_manifest = root / "manifest.json"
        validation = root / "cpu_validation_passed.json"
        if validation.exists():
            if not existing_manifest.exists() or json.loads(existing_manifest.read_text()) != manifest:
                raise ValueError("The existing physical holdout uses a different frozen protocol")
            data = json.loads(validation.read_text())
            if (not data.get("cpu_tests_passed") or data.get("model") != model or
                data.get("fixed_value_learning_rate") != manifest["fixed_value_learning_rate"]):
                raise ValueError("The existing physical holdout validation changed")
            return data
        if (root / "submission.json").exists() or (root / "cases").exists():
            raise ValueError("A submitted holdout has no valid preparation record")
        runtime = root / "runtime"
        if runtime.exists():
            # A failed CPU validation may be transient. Keep the frozen files
            # and re-run their checks only when the full protocol is intact.
            return validate_frozen(root, python, model, manifest)
        runtime.mkdir()
        frozen = runtime / "repo"
        frozen.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
        for folder in ("src", "experiments", "tests", "examples"):
            shutil.copytree(REPO / folder, frozen / folder, ignore=ignore)
        source = workspace / "query_cartridges_rollout32_olmo_dev_20261104/run.py"
        atomic(runtime / "rollout_driver.py", physical_driver(source.read_text()))
        shutil.copy2(development / "olmo_profile.json", runtime / "olmo_profile.json")
        helpers = workspace / "query_cartridges_linked_roll32_dev_20261104/runtime"
        shutil.copytree(helpers / "helpers", runtime / "helpers", ignore=ignore)
        shutil.copytree(helpers / "scorer", runtime / "scorer", ignore=ignore)
        (root / "logs").mkdir(exist_ok=True)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "development_manifest.json", json.dumps(dev_manifest, indent=2) + "\n")
        atomic(root / "development_snapshot.json", json.dumps(dev_report, indent=2) + "\n")
        atomic(root / "report.py", report_stub(workspace, root))
        atomic(root / "job.sbatch", physical_slurm(root, model))
        return validate_frozen(root, python, model, manifest)


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or not re.fullmatch(
        r"query_cartridges_value_lr_physical_(qwen|gpt)_holdout_20261104", root.name):
        raise ValueError("Unexpected physical holdout directory")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        check = json.loads((root / "cpu_validation_passed.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if (not check["cpu_tests_passed"] or check["model"] != manifest["models"][0] or
            check["fixed_value_learning_rate"] != manifest["fixed_value_learning_rate"]):
            raise ValueError("Frozen physical holdout tests are incomplete")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU job cap is full")
            sent = subprocess.run(["sbatch", "--parsable", "--comment=physical-value-lr-holdout",
                str(root / "job.sbatch")], check=True, capture_output=True, text=True, timeout=30)
            job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, model=manifest["models"][0], cases=800,
                        fixed_value_learning_rate=manifest["fixed_value_learning_rate"],
                        cached_scores=True, online_api_calls=0, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "job-root", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("qwen", "gpt"), required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.job_root, args.python, args.model), indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace, args.job_root), indent=2))


if __name__ == "__main__":
    main()
