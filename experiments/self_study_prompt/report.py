#!/usr/bin/env python3
"""Summarize a no-label self-study prompt probe without scheduling queries."""
import argparse
import json
from pathlib import Path
import statistics
import time


def summarize(data):
    if (data["model"] != "qwen" or data["reference_answers_used"] is not False or
        data["online_api_calls"] != 0 or data["training_runtime_changed"] is not False or
        len(data["cases"]) != 10 or not set(data["rows"]).issubset(data["cases"])):
        raise ValueError("The self-study prompt probe protocol changed")
    rows = list(data["rows"].values())
    for row in rows:
        for name in ("base","variant"):
            value = row[name]
            if (value["reference_answers_used"] or value["student_answer_hint_used"] or
                any(type(value[key]) is not int or value[key] < 0 for key in ("generated_tokens","parsed","verified")) or
                value["verified"] > value["parsed"]):
                raise ValueError("Unexpected self-study metadata")
    def stats(name):
        vals = [row[name] for row in rows]
        return dict(n=len(vals),verified_at_least_one=sum(item["verified"]>0 for item in vals),
            verified_exercises=sum(item["verified"] for item in vals),
            invalid_json=sum(bool(item["rejections"].get("invalid_json")) for item in vals),
            median_generated_tokens=statistics.median(item["generated_tokens"] for item in vals) if vals else None)
    return dict(state="complete" if data["complete"] and len(rows)==10 else "partial",
        n=len(rows),expected=10,base=stats("base"),variant=stats("variant"),
        more_verified=sum(row["variant"]["verified"]>row["base"]["verified"] for row in rows),
        fewer_verified=sum(row["variant"]["verified"]<row["base"]["verified"] for row in rows),
        errors=len(data["errors"]),same_selected_teacher_cache=True,
        question_and_reference_answers_not_saved=True,reference_answers_used=False,
        online_api_calls=0,new_gpu_jobs=0,training_runtime_changed=False,
        downstream_accuracy_not_measured=True,updated=time.time())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True);args=parser.parse_args()
    path=args.directory/"qwen.json"
    if not path.exists():
        print("Self-study prompt probe: waiting for an available allocation")
        return
    result=summarize(json.loads(path.read_text()))
    target=args.directory/"latest.json";temporary=target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result,indent=2)+"\n");temporary.replace(target)
    print(json.dumps(result))


if __name__ == "__main__":main()
