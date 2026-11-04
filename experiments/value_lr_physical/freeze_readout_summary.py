#!/usr/bin/env python3
"""Freeze aggregate first-answer transfer diagnostics without case text or IDs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


FIELDS = (
    "kl_full_native_to_selected_native",
    "kl_full_native_to_selected_adapted",
    "kl_full_native_to_full_adapted",
    "kl_selected_native_to_selected_adapted",
)


def summarize(rows):
    if not rows:
        return {"n": 0}
    before = [r["probe"][FIELDS[0]] for r in rows]
    after = [r["probe"][FIELDS[1]] for r in rows]
    full = [r["probe"][FIELDS[2]] for r in rows]
    selected = [r["probe"][FIELDS[3]] for r in rows]
    ratios = [e / d for d, e in zip(full, selected) if d > 1e-8]
    return dict(
        n=len(rows),
        median_full_to_selected_before=statistics.median(before),
        median_full_to_selected_after=statistics.median(after),
        median_full_update_effect=statistics.median(full),
        median_selected_update_effect=statistics.median(selected),
        median_selected_over_full_effect=statistics.median(ratios) if ratios else None,
        effect_ratio_count=len(ratios),
        closer_to_full=sum(b < a for a, b in zip(before, after)),
        farther_from_full=sum(b > a for a, b in zip(before, after)),
        equal_to_full=sum(b == a for a, b in zip(before, after)),
    )


def freeze(report):
    if (report.get("scope") != "fixed 40-case development diagnostic; first answer position only" or
        report.get("metric") != "full-vocabulary first-token KL on the same original question" or
        report.get("references_used_only_posthoc_for_grouping") is not True or
        report.get("model_or_cache_modified") is not False or
        report.get("teacher_answer_in_student_prompt") is not False or
        report.get("online_api_calls") != 0 or set(report.get("models", {})) != {"qwen", "gpt"}):
        raise ValueError("The saved readout diagnostic contract changed")
    qwen_cases = {row["case"] for row in report["models"]["qwen"]["rows"]}
    gpt_cases = {row["case"] for row in report["models"]["gpt"]["rows"]}
    sources = defaultdict(set)
    for case in qwen_cases:
        fields = case.split("-", 2)
        if len(fields) != 3 or fields[0] != "babilong" or fields[1] not in ("8", "16", "32", "64"):
            raise ValueError("Unexpected BABILONG case identity")
        sources[fields[2]].add(fields[1])
    if (len(qwen_cases) != 40 or len(sources) != 10 or
        any(lengths != {"8", "16", "32", "64"} for lengths in sources.values()) or
        not gpt_cases.issubset(qwen_cases)):
        raise ValueError("Expected four lengths for each of ten shared development questions")
    models = {}
    for name, model in report["models"].items():
        rows, unavailable = model["rows"], model["unavailable"]
        if len({row["case"] for row in rows}) != len(rows) or len(rows) + sum(unavailable.values()) != 40:
            raise ValueError("The 40-case diagnostic coverage changed")
        groups = defaultdict(list)
        for row in rows:
            probe = row["probe"]
            if (probe.get("prefix") != "same original question; before first answer token" or
                any(probe.get(key) is not False for key in
                    ("reference_answers_used", "teacher_answer_in_student_prompt", "full_cache_or_model_modified"))):
                raise ValueError("The readout changed the inference contract")
            if any(type(probe.get(key)) not in (int, float) or not math.isfinite(probe[key]) or probe[key] < 0
                   for key in FIELDS):
                raise ValueError("The recorded KL must be finite and nonnegative")
            if row["whole"] not in (0, 1) or row["selected"] not in (0, 1):
                raise ValueError("The posthoc scores must be binary")
            group = ("both_correct" if row["whole"] and row["selected"] else
                     "whole_only" if row["whole"] else "selected_only" if row["selected"] else "both_wrong")
            if row["group"] != group:
                raise ValueError("The posthoc view grouping changed")
            groups[group].append(row)
        models[name] = dict(overall=summarize(rows), groups={k: summarize(v) for k, v in sorted(groups.items())},
                            unavailable=unavailable)
    return dict(scope=report["scope"], metric=report["metric"], source_questions=10,
                context_lengths_per_source=4, model_or_cache_modified=False,
                references_used_only_posthoc_for_grouping=True,
                teacher_answer_in_student_prompt=False, online_api_calls=0, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = freeze(json.loads(args.source.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for name, entry in data["models"].items():
        print(name, json.dumps(entry["overall"]))


if __name__ == "__main__":
    main()
