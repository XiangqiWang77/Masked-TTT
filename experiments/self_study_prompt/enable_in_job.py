#!/usr/bin/env python3
"""Attach a bounded optional prompt probe to the existing Qwen job."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from experiments.kv_only_cartridge.enable_in_job_probe import atomic_directory

REPO=Path(__file__).resolve().parents[2]
JOB="query_cartridges_proper_where_qwen_dev_20261104"
MODULES=("_layout.py","cache.py","model.py","request.py","runtime.py","selection.py",
         "frozen_scores.py","self_study.py","self_study_runtime.py","teacher.py")


def enable(workspace,systemd_root):
    root=Path(workspace).resolve()/JOB
    submission=json.loads((root/"submission.json").read_text())
    manifest=json.loads((root/"manifest.json").read_text())
    driver=root/"runtime/rollout_driver.py"
    if (submission["model"] != "qwen" or submission["gpus"] != 1 or
        manifest["selection_policy"] != "decision_proper_where_v1" or
        manifest["models"] != ["qwen"] or manifest["reference_answers_used_in_inference"] is not False or
        "precision=ROOT/'disabled_optional_probe'" not in driver.read_text() or
        "driver.run_if_needed(engine,model,identity)" not in driver.read_text()):
        raise ValueError("The submitted one-GPU job has no compatible optional probe")
    frozen=root/"runtime/repo"
    for name in MODULES:
        old=frozen/"src/masked_ttt"/name;new=REPO/"src/masked_ttt"/name
        if not old.is_file() or old.read_bytes() != new.read_bytes():
            raise ValueError("The frozen supporting runtime changed: "+name)
    for name in ("experiments/babilong_linked_selection/proper_where_selector.py",
                 "experiments/babilong_linked_selection/question_overlap_selector.py"):
        if not (frozen/name).is_file() or (frozen/name).read_bytes() != (REPO/name).read_bytes():
            raise ValueError("The frozen selection implementation changed")
    destination=root/"disabled_optional_probe"
    protocol=dict(enabled=True,model="qwen",cases=10,source_questions=5,context_lengths=[8192,32768],
        max_seconds_per_allocation=360,student_or_teacher_weights_updated=False,reference_answers_used=False,
        online_api_calls=0,new_gpu_jobs=0,training_runtime_changed=False)
    files={"driver.py":(REPO/"experiments/self_study_prompt/probe.py").read_bytes(),
           "report.py":(REPO/"experiments/self_study_prompt/report.py").read_bytes(),
           "ready.json":(json.dumps(protocol,indent=2)+"\n").encode()}
    atomic_directory(destination,files)
    dropin=Path(systemd_root)/"query-cartridges-proper-where-dev.service.d"
    dropin.mkdir(parents=True,exist_ok=True)
    target=dropin/"self-study-probe.conf"
    content=("[Service]\nExecStartPost=/usr/bin/python3 "+str(destination/"report.py")+
             " --directory "+str(destination)+"\n")
    if target.exists() and target.read_text()!=content:
        raise ValueError("An existing optional report uses a different protocol")
    if not target.exists():
        fd,temp=tempfile.mkstemp(prefix=".self-study-",dir=dropin)
        try:
            with os.fdopen(fd,"w") as stream:stream.write(content)
            os.replace(temp,target)
        finally:
            Path(temp).unlink(missing_ok=True)
    return dict(enabled=True,job_id=submission["job_id"],optional_probe=str(destination),
                model="qwen",cases=10,new_gpu_jobs=0,training_runtime_changed=False,
                note="Runs only on a future allocation that reaches the optional preflight hook.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--systemd-root",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(enable(args.workspace,args.systemd_root),indent=2))


if __name__ == "__main__":main()
