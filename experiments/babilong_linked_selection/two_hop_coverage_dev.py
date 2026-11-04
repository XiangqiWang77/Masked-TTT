#!/usr/bin/env python3
"""Audit an unlabeled two-hop selector on fixed development support spans."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/"src"),str(REPO)]
from experiments.babilong_linked_selection import question_overlap_dev as base
from experiments.babilong_linked_selection.two_hop_selector import two_hop


def collect(workspace):
    """Use the established 200-case coverage audit without altering its source."""
    old=base.question_overlap
    def candidate(document,question,spans,records,fraction,*,mode):
        if mode=="mix":return two_hop(document,question,spans,records,fraction)
        return old(document,question,spans,records,fraction,mode=mode)
    base.question_overlap=candidate
    try:data=base.collect(Path(workspace))
    finally:base.question_overlap=old
    if data["cases"]!=200 or data["source_questions"]!=50 or data["fraction"]!=.1:
        raise ValueError("The fixed development coverage protocol changed")
    groups={key:dict(entity25=value["entity25"],two_hop=value["mix"])
            for key,value in data["groups"].items()}
    return dict(cases=200,source_questions=50,fraction=.1,groups=groups,
        decision=base.decision(groups["all"]["two_hop"]),
        graph_designed_after_previous_development_analysis=True,
        support_and_answers_used_only_posthoc=True,selection_accepts_no_labels=True,
        accuracy_of_new_selection_not_measured=True,online_api_calls=0,new_model_calls=0,
        prior_other_selection_cohort_analysis_seen=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    data=collect(args.workspace.resolve());args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n");temporary.replace(args.output)
    print(json.dumps(dict(overall=data["groups"]["all"],decision=data["decision"]),indent=2))


if __name__=="__main__":main()
