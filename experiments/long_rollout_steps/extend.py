#!/usr/bin/env python3
"""Extend the completed 56-case SD checkpoint pilot to 200 development cases.

The 56 results are reused unchanged.  One new one-GPU job evaluates only the
remaining cases; the 800 source-disjoint cases remain untouched.  This module
uses labels only through the existing post-generation report, never inference.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.prepare import atomic
from experiments.long_rollout_steps.prepare import pilot_slurm, report_stub
from experiments.long_rollout_steps.report import validate_manifest


def isolate_gpu_preflight(script):
    old = '-m pytest -q "$TASK_REPO/tests/test_cuda_materialization.py"'
    new = ('-m pytest --rootdir "$TASK_REPO" --confcutdir "$TASK_REPO" '
           '-q "$TASK_REPO/tests/test_cuda_materialization.py"')
    if script.count(old) != 1:
        raise ValueError("Unexpected GPU preflight command")
    return script.replace(old, new, 1)


def extension_manifest(pilot, report):
    """Require the complete matched pilot before reusing any of its output."""
    if validate_manifest(pilot) != 56:
        raise ValueError("Expected the 56-case source pilot")
    validate_partition(pilot["development_case_ids"], pilot["holdout_case_ids"])
    if (report["expected_per_model"] != 56 or report["metric"] != "answer_focused_lexical_v2" or
        not report["cached_scores"] or report["online_api_calls"] != 0 or
        not report["evaluation_answers_used_only_posthoc"] or report["errors"] or report["invalid"]):
        raise ValueError("The completed pilot report must be clean and use the same scorer")
    if not pilot["models"] or set(pilot["models"]) != set(report["models"]):
        raise ValueError("The pilot model set changed")
    expected = set(pilot["case_ids"])
    for model in pilot["models"]:
        rows = report["models"][model]["rows"]
        if len(rows) != 56 or {row["case"] for row in rows} != expected:
            raise ValueError("The pilot has not completed all matched cases for " + model)
    result = dict(pilot, scope="Original-method SD step checkpoints: 200-case development",
        mode="development", case_count=200, case_ids=list(pilot["development_case_ids"]),
        carried_forward_cases_per_model=56, new_cases_per_model=144,
        configuration_selected_from_pilot=False, holdout_evaluated_at_selection=False)
    validate_manifest(result)
    return result


def _copy_atomic(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise ValueError("A reused pilot case differs from its source: " + str(target))
        return
    fd, temporary = tempfile.mkstemp(prefix=".result-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, out)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare(workspace, pilot_root, root, *, pilot_report=None):
    workspace, pilot_root, root = workspace.resolve(), pilot_root.resolve(), root.resolve()
    if (pilot_root.parent != workspace or pilot_root.name != "query_cartridges_longsteps_pilot_20261104" or
        root.parent != workspace or root.name != "query_cartridges_longsteps_200dev_20261104"):
        raise ValueError("Use the dedicated pilot and 200-case development directories")
    if not (pilot_root / "finished.json").exists():
        raise RuntimeError("The source pilot has not completed")
    pilot = json.loads((pilot_root / "manifest.json").read_text())
    report = (json.loads((pilot_root / "latest.json").read_text())
              if pilot_report is None else pilot_report)
    manifest = extension_manifest(pilot, report)
    root.mkdir(exist_ok=True)
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "manifest.json").exists():
            existing = json.loads((root / "manifest.json").read_text())
            if existing != manifest:
                raise ValueError("The existing extension has a different protocol")
        if (root / "submission.json").exists():
            if not (root / "prepare_complete.json").is_file():
                raise ValueError("A submitted extension has no preparation record")
            return manifest
        runtime = root / "runtime"
        if not runtime.exists():
            stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
            try:
                staged = stage / "runtime"
                shutil.copytree(pilot_root / "runtime", staged,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
                # Only the reporter changes; the frozen model, optimizer and
                # checkpoint inference code are copied from the 56-case run.
                shutil.copy2(REPO / "experiments/long_rollout_steps/report.py",
                             staged / "repo/experiments/long_rollout_steps/report.py")
                atomic(staged / "extension_provenance.json", json.dumps(
                    dict(source=str(pilot_root), case_inference_code="copied from completed pilot",
                         reporter="200-case compatible", online_api_calls=0), indent=2) + "\n")
                os.replace(staged, runtime)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        elif not (runtime / "extension_provenance.json").exists():
            raise ValueError("The existing extension runtime lacks its provenance record")
        for model in manifest["models"]:
            for case in pilot["case_ids"]:
                _copy_atomic(pilot_root / "cases" / model / case / "result.json",
                             root / "cases" / model / case / "result.json")
        (root / "logs").mkdir(exist_ok=True)
        atomic(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic(root / "report.py", report_stub(workspace, root))
        atomic(root / "job.sbatch", isolate_gpu_preflight(pilot_slurm(root)))
        subprocess.run([sys.executable, str(root / "report.py")], check=True, timeout=120,
                       capture_output=True, text=True)
        reused = json.loads((root / "latest.json").read_text())
        if reused["errors"] or reused["invalid"] or any(
            reused["models"][model]["overall"]["n"] < 56 for model in manifest["models"]):
            raise ValueError("Reused pilot results failed the extended reporter")
        atomic(root / "prepare_complete.json", json.dumps(
            dict(source=str(pilot_root), reused_per_model=56, remaining_per_model=144,
                 same_inference_code=True, online_api_calls=0, created=time.time()), indent=2) + "\n")
    return manifest


def submit(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_longsteps_200dev_20261104":
        raise ValueError("Unexpected development directory")
    sys.path.insert(0, str(workspace))
    from gpu_slot_budget import submission_slot
    with (root / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "submission.json"
        if target.exists():
            return json.loads(target.read_text())
        if not (root / "prepare_complete.json").is_file():
            raise ValueError("The copied pilot must pass the extended reporter first")
        manifest = json.loads((root / "manifest.json").read_text())
        if validate_manifest(manifest) != 200:
            raise ValueError("The development manifest is incomplete")
        with submission_slot() as available:
            if not available:
                raise RuntimeError("The shared 12-GPU job cap is full; retry on the next interval")
            sent = subprocess.run(["sbatch", "--parsable", "--job-name=qc-longsteps-200",
                "--comment=long-sd-checkpoint-development", str(root / "job.sbatch")],
                check=True, capture_output=True, text=True, timeout=30)
            job = sent.stdout.strip().split(";", 1)[0]
            if not job.isdigit():
                raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job, gpus=1, models=manifest["models"], cases_per_model=200,
                        cases_reused_per_model=56, new_cases_per_model=144, cached_scores=True,
                        online_api_calls=0, created=time.time())
            atomic(target, json.dumps(data, indent=2) + "\n")
            return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "pilot-root", "job-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.workspace, args.pilot_root, args.job_root)
    print(json.dumps(dict(prepared=manifest["case_count"], reused=56), indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace, args.job_root), indent=2))


if __name__ == "__main__":
    main()
