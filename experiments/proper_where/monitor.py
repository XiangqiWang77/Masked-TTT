#!/usr/bin/env python3
"""Hourly Qwen report and conditional source-disjoint holdout."""
import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/"src"),str(REPO)]
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.prepare import atomic
from experiments.proper_where.prepare import DEV,HOLDOUT,batch_script,enable_monitor,report_stub,submit,validate_frozen
from experiments.proper_where.report import render,summarize


def development_decision(report):
    stats = report["overall"]
    if (report["mode"] != "development" or report["expected"] != 200 or stats["n"] != 200 or
        report["metric"] != "answer_focused_lexical_v2" or report["errors"] or
        report["invalid"] or report["unmatched"] or
        stats["changed_selection"] != report["expected_development_changed_selections"]):
        return dict(eligible=False,reason="development_incomplete_or_unmatched")
    a=stats["accuracy"]
    differences={key:a["cartridge"]-a[key] for key in ("whole","zero","teacher","old_cartridge")}
    passed=all(value>0 for value in differences.values())
    return dict(eligible=passed,reason="beats_all_paired_controls" if passed else "does_not_beat_all_paired_controls",
                delta_pp=differences,changed_selections=stats["changed_selection"])


def prepare_holdout(workspace,development,report):
    source=json.loads((development/"manifest.json").read_text())
    validate_partition(source["development_case_ids"],source["holdout_case_ids"])
    choice=development_decision(report)
    if not choice["eligible"]:
        atomic(development/"no_eligible_holdout.json",json.dumps(choice,indent=2)+"\n")
        return None,choice
    root=workspace/HOLDOUT;root.mkdir(exist_ok=True)
    manifest=dict(source,mode="holdout",case_count=800,case_ids=list(source["holdout_case_ids"]),
        scope="Qwen question-guided selection, original SD, source-disjoint holdout",
        development_decision=choice,development_used_for_configuration_selection=True,
        holdout_evaluated_at_selection=False)
    with (root/"promotion.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/"runtime").exists():
            if not (root/"manifest.json").exists() or json.loads((root/"manifest.json").read_text()) != manifest:
                raise ValueError("The existing holdout uses another fixed protocol")
        else:
            shutil.copytree(development/"runtime",root/"runtime",
                            ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
            (root/"logs").mkdir(exist_ok=True)
            atomic(root/"manifest.json",json.dumps(manifest,indent=2)+"\n")
            atomic(root/"development_snapshot.json",json.dumps(report,indent=2)+"\n")
            atomic(root/"report.py",report_stub(workspace,root))
            atomic(root/"job.sbatch",batch_script(root))
    return root,choice


def combine(manifest,dev,held):
    development,holdout=dev["rows"],held["rows"]
    if (dev["metric"] != held["metric"] or dev["errors"] or dev["invalid"] or dev["unmatched"] or
        held["errors"] or held["invalid"] or held["unmatched"] or len(development)!=200 or len(holdout)!=800):
        raise ValueError("A complete, valid and paired 200/800 split is required")
    validate_partition([row["case"] for row in development],[row["case"] for row in holdout])
    if ({row["case"] for row in development} != set(manifest["development_case_ids"]) or
        {row["case"] for row in holdout} != set(manifest["holdout_case_ids"])):
        raise ValueError("The fixed case identities changed")
    groups=defaultdict(list)
    for row in development+holdout:
        groups["task:"+row["task"]].append(row);groups["length:"+str(row["length"])].append(row)
        groups["selection_changed:"+str(row["selection_changed"]).lower()].append(row)
    inspected=set(manifest["qualitative_case_sources"])
    unviewed=[row for row in holdout if row["case"].split("-",2)[2] not in inspected]
    if len(unviewed)!=792:raise ValueError("Unexpected previously viewed source exclusion")
    return dict(mode="combined",model="qwen",expected=1000,metric=held["metric"],
        overall=summarize(development+holdout),development=summarize(development),holdout=summarize(holdout),
        holdout_excluding_two_previously_viewed_sources=summarize(unviewed),
        groups={key:summarize(items) for key,items in sorted(groups.items())},
        development_used_to_design_selector=True,prior_other_selection_cohort_analysis_seen=True)


def advance(workspace,root,python):
    workspace,root=workspace.resolve(),root.resolve()
    if root.parent != workspace or root.name not in (DEV,HOLDOUT):
        raise ValueError("Unexpected Qwen question-overlap job directory")
    subprocess.run([sys.executable,str(root/"report.py")],check=True,capture_output=True,text=True,timeout=120)
    if not (root/"finished.json").exists():
        sys.path.insert(0,str(workspace))
        from cartridge_job_watch import record_terminal
        terminal=record_terminal(root)
        return dict(state="terminal_without_complete_results" if terminal else "in_progress",scheduler_state=terminal)
    report=json.loads((root/"latest.json").read_text())
    if root.name==DEV:
        held,choice=prepare_holdout(workspace,root,report)
        if held is None:return dict(state="not_eligible",decision=choice)
        if not (held/"cpu_validation_passed.json").exists():validate_frozen(held,python)
        try:sent=submit(workspace,held)
        except RuntimeError as exc:
            if str(exc)=="The shared 12-GPU job cap is full":return dict(state="waiting_slot")
            raise
        enable_monitor(workspace,held,python)
        return dict(state="submitted",job_id=sent["job_id"])
    manifest=json.loads((root/"manifest.json").read_text())
    dev=json.loads((root/"development_snapshot.json").read_text())
    report=combine(manifest,dev,report)
    atomic(root/"combined_1000.json",json.dumps(report,indent=2)+"\n")
    atomic(root/"combined_1000.md",render(report)+
        "The selector was designed using 200 development cases. The 800 holdout cases have disjoint "
        "source questions; two holdout sources and previous cohort results had already been inspected.\n")
    return dict(state="complete",report=str(root/"combined_1000.md"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("workspace","job-root","python"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--timer",required=True)
    args=parser.parse_args();root=args.job_root.resolve()
    try:result=advance(args.workspace,root,args.python)
    except Exception as exc:
        atomic(root/"monitor_status.json",json.dumps(dict(state="monitor_error",error_type=type(exc).__name__,
               updated=time.time()),indent=2)+"\n");raise
    atomic(root/"monitor_status.json",json.dumps(dict(result,updated=time.time()),indent=2)+"\n")
    print(json.dumps(result,default=str))
    if result["state"] in ("terminal_without_complete_results","not_eligible","submitted","complete"):
        subprocess.run(["systemctl","--user","stop",args.timer],check=True,capture_output=True,text=True)


if __name__ == "__main__":main()
