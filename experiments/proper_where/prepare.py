#!/usr/bin/env python3
"""Freeze and submit one Qwen question-guided selection development job."""
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

DEV = "query_cartridges_proper_where_qwen_dev_20261104"
HOLDOUT = "query_cartridges_proper_where_qwen_holdout_20261104"
CPU_TESTS = ["tests/test_proper_where.py", "tests/test_question_overlap_selector.py",
             "tests/test_query_head_runner.py", "tests/test_public_linked_selection.py",
             "tests/test_cli.py", "tests/test_frozen_scores.py", "tests/test_batched_sd.py",
             "tests/test_materialization_equivalence.py", "tests/test_kv_only_cartridge.py"]


def make_manifest(rollout, linked, profile, coverage):
    dev, held = linked["case_ids"], linked["holdout_case_ids"]
    validate_partition(dev, held)
    candidate = coverage["groups"]["all"]["proper_where"]
    if (dev != rollout["case_ids"] or held != rollout["holdout_case_ids"] or
        linked["model_protocols"] != rollout["model_protocols"] or
        linked["selection_policy"] != "decision_entity_link_v1" or linked["entity_reserve_fraction"] != .25 or
        linked["keep_fraction"] != .1 or linked["selected_transfer_weight"] != 0 or
        linked["reference_answers_used_in_inference"] is not False or linked["online_api_calls"] != 0 or
        profile["student_rollout_tokens"] != 8 or profile["steps"] != 2 or profile["max_answer_steps"] != 8 or
        profile["gain_learning_rate"] != .01 or profile["value_learning_rate"] != .0001 or
        profile["answer_KL_stop"] != .5 or profile["self_question_count"] != 2 or
        coverage["cases"] != 200 or coverage["source_questions"] != 50 or coverage["fraction"] != .1 or
        coverage["selection_accepts_no_labels"] is not True or coverage["support_and_answers_used_only_posthoc"] is not True or
        coverage["accuracy_of_new_selections_not_measured"] is not True or coverage["online_api_calls"] != 0 or
        coverage["new_model_calls"] != 0 or
        not coverage["decisions"]["proper_where"]["eligible_for_model_test"] or
        "proper_where" not in coverage["variants_designed_after_reviewing_development_results"] or
        candidate["net_complete"] < 5 or candidate["full_only_newly_complete"] < 3 or
        candidate["teacher_only_lost_complete"] > 2):
        raise ValueError("The original SD baseline or the development coverage evidence changed")
    return dict(rollout,scope="Qwen cached Decision with proper-name question overlap; original OLMo-style SD",
        mode="development",case_count=200,case_ids=list(dev),development_case_ids=list(dev),
        holdout_case_ids=list(held),models=["qwen"],selection_policy="decision_proper_where_v1",
        question_overlap_policy="proper_where",entity_reserve_fraction=.25,student_rollout_tokens=8,
        expected_development_changed_selections=candidate["changed_selections"],
        fixed_value_learning_rate=.0001,gain_head_mode="kv",steps=2,max_answer_steps=8,
        holdout_evaluated_at_selection=False,development_used_to_design_selector=True,
        prior_other_selection_cohort_analysis_seen=True,max_gpu_hours=36,online_api_calls=0,
        development_gate="Beat own full, selected zero, Teacher, and paired old cartridge on 200 cases",
        holdout_target="At least +3 percentage points versus own ICL Teacher")


def batch_script(root):
    script = slurm_script(root)
    name = "qc-proper-where-hold" if root.name == HOLDOUT else "qc-proper-where-dev"
    expected = "qc-gqa-gain-hold" if root.name == HOLDOUT else "qc-gqa-gain-dev"
    changes = {expected:name,"gqa_head_gain/execute.py":"proper_where/execute.py",
               "#SBATCH --time-min=00:30:00":"#SBATCH --time-min=00:20:00",
               "#SBATCH --mem=128G":"#SBATCH --mem=64G"}
    for old,new in changes.items():
        if script.count(old) != 1:
            raise ValueError("Unexpected one-GPU batch script: "+old)
        script = script.replace(old,new,1)
    return isolate_gpu_preflight(script)


def report_stub(workspace,root):
    refs = workspace.parent/"chunked_ntp_query_only_20261029/evaluation_references.json"
    baseline = workspace/("query_cartridges_linked_selection_20261104" if root.name == DEV
                         else "query_cartridges_linked_holdout_20261104")
    arguments = ["--manifest",str(root/"manifest.json"),"--candidate-root",str(root),
        "--baseline-root",str(baseline),"--references",str(refs),
        "--scorer-dir",str(root/"runtime/scorer"),"--output",str(root/"latest")]
    return ("#!/usr/bin/env python3\nimport fcntl\nfrom pathlib import Path\nimport sys\n"
            f"sys.path.insert(0, {str(root/'runtime/repo')!r})\n"
            "from experiments.proper_where.report import main\n"
            f"sys.argv = [sys.argv[0], *{arguments!r}]\n"
            f"with Path({str(root/'report.lock')!r}).open('a+') as lock:\n"
            "    fcntl.flock(lock, fcntl.LOCK_EX)\n    main()\n")


def validate_frozen(root,python):
    frozen = root/"runtime/repo"
    env = dict(os.environ,PYTHONPATH="src:tests:.",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    check = subprocess.run([str(python),"-m","pytest","--rootdir",str(frozen),
        "--confcutdir",str(frozen),"-q",*CPU_TESTS],cwd=frozen,env=env,
        text=True,capture_output=True,timeout=180)
    matched = re.search(r"\b(\d+) passed\b",check.stdout)
    if check.returncode or not matched:
        raise RuntimeError("Frozen question-overlap checks failed:\n"+check.stdout+check.stderr)
    data = dict(cpu_tests_passed=True,passed_count=int(matched.group(1)),tests=CPU_TESTS,
                gpu_preflight_required=True,online_api_calls=0,created=time.time())
    atomic(root/"cpu_validation_passed.json",json.dumps(data,indent=2)+"\n")
    return data


def prepare(workspace,root,python):
    workspace,root = workspace.resolve(),root.resolve()
    if root.parent != workspace or root.name != DEV:
        raise ValueError("Use the dedicated Qwen question-overlap directory")
    rollout_root = workspace/"query_cartridges_rollout32_olmo_dev_20261104"
    linked_root = workspace/"query_cartridges_linked_selection_20261104"
    profile_root = workspace/"query_cartridges_value_lr_20261104"
    coverage_path = REPO/"docs/analysis/question-overlap-coverage-dev-20261104/summary.json"
    coverage = json.loads(coverage_path.read_text())
    manifest = make_manifest(json.loads((rollout_root/"manifest.json").read_text()),
        json.loads((linked_root/"manifest.json").read_text()),
        json.loads((profile_root/"olmo_profile.json").read_text()),coverage)
    root.mkdir(exist_ok=True)
    with (root/"submission.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/"submission.json").exists() or (root/"cases").exists():
            raise ValueError("Do not change a submitted or evaluated job")
        runtime = root/"runtime"
        if runtime.exists():
            if not (root/"manifest.json").is_file() or json.loads((root/"manifest.json").read_text()) != manifest:
                raise ValueError("The frozen protocol changed")
        else:
            frozen = runtime/"repo";frozen.mkdir(parents=True)
            ignore = shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc")
            for folder in ("src","experiments","tests","examples"):
                shutil.copytree(REPO/folder,frozen/folder,ignore=ignore)
            atomic(runtime/"rollout_driver.py",physical_driver((rollout_root/"run.py").read_text()))
            shutil.copy2(profile_root/"olmo_profile.json",runtime/"olmo_profile.json")
            shared = workspace/"query_cartridges_linked_roll32_dev_20261104/runtime"
            shutil.copytree(shared/"helpers",runtime/"helpers",ignore=ignore)
            shutil.copytree(shared/"scorer",runtime/"scorer",ignore=ignore)
            (root/"logs").mkdir(exist_ok=True)
            atomic(root/"manifest.json",json.dumps(manifest,indent=2)+"\n")
            atomic(root/"coverage_development_snapshot.json",json.dumps(coverage,indent=2)+"\n")
            atomic(root/"report.py",report_stub(workspace,root))
            atomic(root/"job.sbatch",batch_script(root))
        return validate_frozen(root,python)


def submit(workspace,root):
    workspace,root = workspace.resolve(),root.resolve()
    if root.parent != workspace or root.name not in (DEV,HOLDOUT):
        raise ValueError("Unexpected Qwen question-overlap directory")
    sys.path.insert(0,str(workspace))
    from gpu_slot_budget import submission_slot
    with (root/"submission.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        target = root/"submission.json"
        if target.exists():return json.loads(target.read_text())
        if not json.loads((root/"cpu_validation_passed.json").read_text())["cpu_tests_passed"]:
            raise ValueError("Frozen CPU validation has not passed")
        with submission_slot() as available:
            if not available:raise RuntimeError("The shared 12-GPU job cap is full")
            sent = subprocess.run(["sbatch","--parsable","--comment=qwen-proper-where-olmo-sd",
                str(root/"job.sbatch")],check=True,capture_output=True,text=True,timeout=30)
            job = sent.stdout.strip().split(";",1)[0]
            if not job.isdigit():raise ValueError("Unexpected scheduler response")
            data = dict(job_id=job,gpus=1,model="qwen",cases=200 if root.name == DEV else 800,
                        cached_scores=True,online_api_calls=0,created=time.time())
            atomic(target,json.dumps(data,indent=2)+"\n");return data


def enable_monitor(workspace,root,python):
    unit = Path(os.environ.get("XDG_CONFIG_HOME",str(Path.home()/".config")))/"systemd/user"
    unit.mkdir(parents=True,exist_ok=True)
    stage = "holdout" if root.name == HOLDOUT else "dev"
    timer = f"query-cartridges-proper-where-{stage}.timer"
    service = ("[Unit]\nDescription=Report Qwen question-guided selection and original SD\n\n"
        "[Service]\nType=oneshot\n"
        "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
        f"ExecStart=/usr/bin/python3 {root}/runtime/repo/experiments/proper_where/monitor.py "
        f"--workspace {workspace} --job-root {root} --python {python} --timer {timer}\n")
    timer_text = ("[Unit]\nDescription=Update Qwen question-guided study hourly\n\n"
        "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
        "[Install]\nWantedBy=timers.target\n")
    atomic(unit/f"query-cartridges-proper-where-{stage}.service",service)
    atomic(unit/timer,timer_text)
    subprocess.run(["systemctl","--user","daemon-reload"],check=True,capture_output=True,text=True)
    subprocess.run(["systemctl","--user","enable","--now",timer],check=True,capture_output=True,text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace","job-root","python"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--submit",action="store_true")
    args=parser.parse_args()
    print(json.dumps(prepare(args.workspace,args.job_root,args.python),indent=2))
    if args.submit:
        print(json.dumps(submit(args.workspace,args.job_root),indent=2))
        enable_monitor(args.workspace.resolve(),args.job_root.resolve(),args.python)


if __name__ == "__main__":main()
