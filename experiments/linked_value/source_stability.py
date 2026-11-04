#!/usr/bin/env python3
"""Analyze completed GPT paired changes by source and evidence coverage.

This is descriptive post-generation analysis. It does not call a model, use
references during inference or export questions, answers or case identities.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.linked_value.freeze_development import collect as validate_development


STATUSES = ("complete", "incomplete", "ambiguous", "unavailable")


def source_pairs(rows, base):
    available = [row for row in rows if row[base] is not None]
    if any(type(row[base]) is not int or row[base] not in (0, 1) or
           type(row["cartridge"]) is not int or row["cartridge"] not in (0, 1) for row in available):
        raise ValueError("Expected paired binary post-generation outcomes")
    groups = defaultdict(list)
    for row in available:
        groups[row["source"]].append(row)
    counts = Counter()
    changes = Counter()
    for group in groups.values():
        good = sum(not row[base] and row["cartridge"] for row in group)
        bad = sum(row[base] and not row["cartridge"] for row in group)
        label = "mixed" if good and bad else "improved_only" if good else "regressed_only" if bad else "unchanged"
        counts[label] += 1
        changes[str(good + bad)] += 1
    good = sum(not row[base] and row["cartridge"] for row in available)
    bad = sum(row[base] and not row["cartridge"] for row in available)
    return dict(n=len(available), source_questions=len(groups), new_correct=good, new_wrong=bad,
        delta_pp=100 * (good-bad) / len(available) if available else None,
        source_outcomes={label: counts[label] for label in ("improved_only", "regressed_only", "mixed", "unchanged")},
        changed_lengths_per_source={str(i): changes[str(i)] for i in range(5)})


def summarize(rows):
    if any(row["coverage"] not in STATUSES or row["whole"] not in (0, 1) or row["teacher"] not in (0, 1)
           for row in rows):
        raise ValueError("Invalid post-generation answer or support coverage")
    if any(type(row["verified"]) is not int or row["verified"] not in (0, 1, 2) for row in rows):
        raise ValueError("Invalid source-verified practice count")
    whole_only = [row for row in rows if row["whole"] and not row["teacher"]]
    return dict(n=len(rows), source_questions=len({row["source"] for row in rows}),
        verified_at_least_one=sum(row["verified"] > 0 for row in rows),
        coverage={status: sum(row["coverage"] == status for row in rows) for status in STATUSES},
        versus={base: source_pairs(rows, base) for base in ("teacher", "zero", "baseline", "whole")},
        full_correct_teacher_wrong=dict(n=len(whole_only),
            source_questions=len({row["source"] for row in whole_only}),
            cartridge_correct=sum(row["cartridge"] for row in whole_only),
            zero_correct=sum(row["zero"] for row in whole_only),
            versus_zero=source_pairs(whole_only, "zero"),
            coverage={status: sum(row["coverage"] == status for row in whole_only) for status in STATUSES}))


def collect(root, coverage_path):
    validated = validate_development(root)
    report = json.loads((root / "latest.json").read_text())
    coverage = json.loads(coverage_path.read_text())
    annotations = {row["case"]: row for row in coverage["rows"]}
    if (len(coverage["rows"]) != 200 or len(annotations) != 200 or
        set(annotations) != {row["case"] for row in report["rows"]} or
        coverage["answers_and_annotations_used_only_posthoc"] is not True or coverage["online_api_calls"] != 0):
        raise ValueError("The post-generation annotated cohort differs")
    rows, lengths = [], defaultdict(set)
    for row in report["rows"]:
        key = row["case"]
        source = key.split("-", 2)[2]
        annotated = annotations[key]
        result = json.loads((root / "cases/gpt" / key / "result.json").read_text())
        entity = annotated["outcomes"]["entity10"]
        selected = sorted(index for group in result["selection"]["selected_original_chunks"] for index in group)
        if (annotated["task"] != row["task"] or annotated["length"] != row["length"] or
            selected != entity["selected"] or entity["status"] not in STATUSES):
            raise ValueError("The scored answer and annotated source selection differ")
        lengths[source].add(int(key.split("-", 2)[1]))
        rows.append(dict(source=source, task=row["task"], length=row["length"], coverage=entity["status"],
            whole=row["whole"], teacher=row["teacher"], zero=row["zero"],
            cartridge=row["cartridge"], baseline=row["baseline"], verified=result["self_study"]["verified"]))
    if len(lengths) != 50 or any(value != {8, 16, 32, 64} for value in lengths.values()):
        raise ValueError("The development cohort is not 50 sources at four lengths")
    return dict(model="GPT-OSS-20B", metric=validated["metric"], cases=200, source_questions=50,
        original_sd_training=True, annotations_and_reference_scores_only_posthoc=True,
        teacher_and_original_cartridge_formats_differ=True, online_api_calls=0, new_model_calls=0,
        coverage_means_annotated_support_fact_spans_covered=True,
        coverage_does_not_prove_facts_were_read_or_all_necessary_context_was_present=True,
        overall=summarize(rows),
        by_verified={label: summarize([row for row in rows if (row["verified"] > 0) == present])
                     for label, present in (("none", False), ("at_least_one", True))},
        by_coverage={status: summarize([row for row in rows if row["coverage"] == status]) for status in STATUSES},
        by_task={task: summarize([row for row in rows if row["task"] == task])
                 for task in sorted({row["task"] for row in rows})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.job_root.resolve(), args.coverage.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["overall"], indent=2))


if __name__ == "__main__":
    main()
