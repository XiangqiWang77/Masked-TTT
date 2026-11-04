#!/usr/bin/env python3
"""Read only the Query Cartridge jobs whose hourly check is due."""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SPECS={
    "rollout_value_qwen":"query_cartridges_rollout_value_qwen_dev_20261104",
    "canonical_pair":"query_cartridges_canonical_pair_dev_20261104",
    "linked_holdout":"query_cartridges_linked_holdout_20261104",
    "linked_value_gpt_holdout":"query_cartridges_linked_value_gpt_holdout_20261104",
    "gain_rate_pair":"query_cartridges_gain_rate_pair_dev_20261104",
    "physical_gpt":"query_cartridges_value_lr_physical_gpt_holdout_20261104",
    "physical_step_gpt":"query_cartridges_step_physical_gpt_holdout_20261104",
    "rollout_value_qwen_holdout":"query_cartridges_rollout_value_qwen_holdout_20261104",
    "canonical_pair_holdout":"query_cartridges_canonical_pair_holdout_20261104",
    "gain_rate_qwen_holdout":"query_cartridges_gain_rate_qwen_holdout_20261104",
    "gain_rate_gpt_holdout":"query_cartridges_gain_rate_gpt_holdout_20261104",
}
PROMOTIONS={"rollout_value_qwen":["rollout_value_qwen_holdout"],
            "canonical_pair":["canonical_pair_holdout"],
            "gain_rate_pair":["gain_rate_qwen_holdout","gain_rate_gpt_holdout"]}
RESOLVED={"submitted","not_eligible","finished_development","terminal_without_complete_results","complete"}


def due_jobs(schedule,now):
    if now.tzinfo is None:
        raise ValueError("The check time must have a timezone")
    jobs=[]
    for name in SPECS:
        raw=schedule.get(name+"_next_check_not_before_utc")
        job=str(schedule.get(name+"_job_id") or "")
        if raw is None or not job.isdigit():continue
        due=datetime.fromisoformat(raw.replace("Z","+00:00"))
        if due.tzinfo is None:raise ValueError("A check deadline lacks a timezone")
        if now>=due:jobs.append((name,job))
    return jobs


def query_scheduler(job,workspace):
    """Do not infer termination from an absent or unavailable queue response."""
    try:
        done=subprocess.run(["squeue","--noheader","--jobs="+job,"--format=%i|%T|%R|%M"],
                            capture_output=True,text=True,timeout=20,check=False)
    except (OSError,subprocess.TimeoutExpired):
        return dict(state="UNKNOWN",reason="queue_query_unavailable",terminal=False)
    if done.returncode:
        return dict(state="UNKNOWN",reason="queue_query_unavailable",terminal=False)
    for line in done.stdout.splitlines():
        fields=line.split("|",3)
        if len(fields)==4 and fields[0]==job:
            return dict(state=fields[1],reason=fields[2],elapsed=fields[3],terminal=False)
    # The existing guard checks Slurm control/accounting and requeue races.
    path=Path(workspace)/"cartridge_job_watch.py"
    spec=importlib.util.spec_from_file_location("hourly_cartridge_job_watch",path)
    if spec is None or spec.loader is None:
        return dict(state="UNKNOWN",reason="terminal_guard_unavailable",terminal=False)
    try:
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        state=module.observed_terminal(job)
    except (OSError,ImportError,AttributeError,ValueError):
        return dict(state="UNKNOWN",reason="terminal_guard_unavailable",terminal=False)
    return dict(state=state or "UNKNOWN",reason="confirmed_terminal" if state else "not_confirmed",
                terminal=bool(state))


def _stats(item):
    if not isinstance(item,dict):return None
    keys=("n","accuracy","versus_teacher","versus_zero","strictly_paired",
          "paired_baseline_accuracy","versus","step8_vs_teacher","step8_vs_zero")
    return {key:item[key] for key in keys if key in item}


def read_summary(workspace,name):
    root=Path(workspace)/SPECS[name];path=root/"latest.json"
    if not path.is_file():return None
    try:data=json.loads(path.read_text())
    except (OSError,ValueError):return dict(status="report_unavailable")
    if (not isinstance(data,dict) or not isinstance(data.get("errors",[]),list) or
        not isinstance(data.get("invalid",[]),list)):
        return dict(status="report_invalid")
    result=dict(report_created_utc=data.get("created_utc"),
        report_modified_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
        expected=data.get("expected",data.get("expected_per_model")),
        metric=data.get("metric"),errors=len(data.get("errors",[])),invalid=len(data.get("invalid",[])))
    if isinstance(data.get("models"),dict):
        result["models"]={model:_stats(value.get("overall")) if isinstance(value,dict) else None
                          for model,value in data["models"].items()}
    elif isinstance(data.get("overall"),dict):
        result["overall"]=_stats(data["overall"])
    return result


def read_probe(workspace,name):
    if name=="rollout_value_qwen":
        path=Path(workspace)/"query_cartridges_reusable_gpu_probe_20261104/qwen_result.json"
        if path.is_file():
            try:data=json.loads(path.read_text())
            except (OSError,ValueError):return dict(status="report_unavailable")
            if not isinstance(data,dict):return dict(status="report_invalid")
            return dict(kind="qwen_reusable_cartridge",passed=data.get("passed"),gpu=data.get("gpu"))
    if name=="gain_rate_pair":
        path=Path(workspace)/"query_cartridges_physical_counterfactual_probe_20261104/latest.json"
        if path.is_file():
            try:data=json.loads(path.read_text())
            except (OSError,ValueError):return dict(status="report_unavailable")
            if not isinstance(data,dict) or not isinstance(data.get("models"),dict):
                return dict(status="report_invalid")
            allowed=("status","gpu","margin_difference_garden_minus_office_context",
                     "full_baseline_mentions_expected_for_both","cartridge_mentions_expected_for_both")
            return dict(kind="physical_counterfactual",models={model:{key:row[key] for key in allowed if key in row}
                for model,row in data["models"].items() if isinstance(row,dict)})
    return None


def register_promotions(workspace,parent,state):
    """Discover holdouts only when their parent is due for its hourly check."""
    registered=[]
    for child in PROMOTIONS.get(parent,[]):
        if state.get(child+"_job_id"):continue
        path=Path(workspace)/SPECS[child]/"submission.json"
        if not path.is_file():continue
        try:info=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if not isinstance(info,dict):continue
        job=str(info.get("job_id") or "");created=info.get("created")
        if not job.isdigit() or type(created) not in (int,float):continue
        try:when=datetime.fromtimestamp(created,timezone.utc)
        except (ValueError,OverflowError,OSError):continue
        state[child+"_job_id"]=job
        state[child+"_last_state"]="SUBMITTED"
        state[child+"_last_submission_utc"]=when.isoformat()
        state[child+"_next_check_not_before_utc"]=(when+timedelta(hours=1)).isoformat()
        registered.append(dict(name=child,job_id=job))
    return registered


def promotion_resolved(workspace,name):
    if name not in PROMOTIONS:return True
    path=Path(workspace)/SPECS[name]/"monitor_status.json"
    if not path.is_file():return False
    try:status=json.loads(path.read_text())
    except (OSError,ValueError):return False
    return isinstance(status,dict) and status.get("state") in RESOLVED


def run(workspace,*,now=None,query=query_scheduler,summary=read_summary,probe=read_probe):
    workspace=Path(workspace).resolve();path=workspace/"monitor/cartridge_user_interval_20261104.json"
    now=datetime.now(timezone.utc) if now is None else now
    lockpath=path.with_suffix(".lock")
    with lockpath.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(path.read_text());due=due_jobs(state,now);observed=[]
        for name,job in due:
            status=query(job,workspace)
            item=dict(name=name,job_id=job,scheduler=status,summary=summary(workspace,name),probe=probe(workspace,name),
                      new_holdouts=register_promotions(workspace,name,state))
            observed.append(item)
            state[name+"_last_state"]=status["state"]
            state[name+"_last_check_utc"]=now.isoformat()
            finished=status.get("terminal") and promotion_resolved(workspace,name)
            state[name+"_next_check_not_before_utc"]=None if finished else (now+timedelta(hours=1)).isoformat()
            result=item["summary"] or {}
            if isinstance(result.get("overall"),dict):
                state[name+"_last_report_cases"]=result["overall"].get("n")
        if observed:
            temporary=path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")
            os.replace(temporary,path)
    return observed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.workspace),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
