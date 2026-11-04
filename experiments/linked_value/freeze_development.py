#!/usr/bin/env python3
"""Freeze completed GPT linked-value development aggregates without case text."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from experiments.gqa_head_gain.report import summarize
from experiments.value_lr_physical.canonical_memory_report import memory_record, summarize as memory_summary


def collect(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "latest.json").read_text())
    finished = json.loads((root / "finished.json").read_text())
    keys = manifest["case_ids"]
    rows = report["rows"]
    if (manifest["mode"] != "development" or manifest["models"] != ["gpt"] or
        manifest["case_count"] != 200 or len(set(keys)) != 200 or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["fixed_value_learning_rate"] != .001 or manifest["student_rollout_tokens"] != 8 or
        manifest["selected_transfer_weight"] != 0 or manifest["reference_answers_used_in_inference"] is not False or
        manifest["online_api_calls"] != 0 or report["metric"] != "answer_focused_lexical_v2" or
        report["mode"] != "development" or report["expected"] != 200 or report["fixed_value_learning_rate"] != .001 or
        report["errors"] or report["invalid"] or len(rows) != 200 or
        {row["case"] for row in rows} != set(keys) or len({row["case"] for row in rows}) != 200 or
        summarize(rows) != report["overall"] or not finished["all_complete"] or finished["completed"]["gpt"] != 200):
        raise ValueError("The completed GPT linked-value development differs from its fixed protocol")
    if report["overall"]["strictly_paired"]["n"] + len(report["unmatched_baselines"]) != 200:
        raise ValueError("The strict matched and unmatched cases do not cover the development set")
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    if {key: summarize(group) for key, group in sorted(groups.items())} != report["groups"]:
        raise ValueError("The task and length aggregates differ from the paired outcomes")
    memory_rows, memory_groups = [], defaultdict(list)
    for row in rows:
        result = json.loads((root / "cases/gpt" / row["case"] / "result.json").read_text())
        if (result["key"] != row["case"] or result["model_short"] != "gpt" or
            result["task"] != row["task"] or result["length"] != row["length"] or
            result["training"]["value_learning_rate"] != .001 or
            result["training"]["gain_learning_rate"] != .01 or
            result["training"]["selected_transfer_weight"] != 0 or
            result["training"]["objective"] != "on-policy KL(teacher || student)" or
            result["selection"]["selection_policy"] != "decision_entity_link_v1" or
            result["selection"]["reference_answers_used"] or result["selection"]["online_api_calls"] != 0):
            raise ValueError("A completed cartridge does not match the fixed GPT method")
        record = memory_record(result)
        if not math.isclose(row["cartridge_percent"], 100 * record["cartridge"] / record["full"], abs_tol=1e-9):
            raise ValueError("The stored cartridge ratio differs from its physical byte counts")
        memory_rows.append(record)
        memory_groups["length:" + str(row["length"])].append(record)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), model="GPT-OSS-20B", cases=200,
        source_questions=50, metric=report["metric"], online_api_calls=0, new_model_calls=0,
        reference_answers_used_only_posthoc=True, teacher_and_original_cartridge_formats_differ=True,
        selected_transfer_weight=0, gain_learning_rate=.01, value_learning_rate=.001,
        original_sd_training=True, physically_released_standalone=True,
        overall=report["overall"], groups=report["groups"],
        unmatched_baseline_count=len(report["unmatched_baselines"]),
        memory=memory_summary(memory_rows),
        memory_by_length={key: memory_summary(group) for key, group in sorted(memory_groups.items())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.job_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["overall"], indent=2))


if __name__ == "__main__":
    main()
