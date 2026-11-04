#!/usr/bin/env python3
"""Aggregate matched task/length groups from already completed BABILONG runs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

ARMS = ("vanilla", "kv_whole_ntp", "kv_selected_ntp", "kv_sd_soft")
LENGTHS = (8192, 16384, 32768, 65536)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("olmo-snapshot", "olmo-storage", "qwen-arms", "qwen-references", "scorer-dir", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.olmo_snapshot.open(newline="") as stream:
        olmo = list(csv.DictReader(stream))
    with args.olmo_storage.open(newline="") as stream:
        storage = list(csv.DictReader(stream))
    if len(olmo) != 1000 or len(storage) != 1000:
        raise ValueError("Expected both complete OLMo files")
    grouped = defaultdict(list)
    for i, (row, mem) in enumerate(zip(olmo, storage)):
        if int(mem["row"]) != i or row["context_tokens"] != mem["context_tokens"]:
            raise ValueError("OLMo storage rows are not aligned with case snapshot")
        ratio = int(mem["cartridge_bytes"]) / int(mem["full_bytes"])
        if not 0 < ratio <= 1:
            raise ValueError("Unexpected KV size ratio")
        grouped[row["task"], int(row["context_tokens"])].append((row, ratio))
    if len(grouped) != 40 or any(len(group) != 25 for group in grouped.values()):
        raise ValueError("Expected 25 OLMo cases per task and context")
    olmo_groups = []
    for (task, length), group in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        correct = {name: sum(int(row[name]) for row, _ in group)
                   for name in ("whole", "teacher", "cartridge")}
        olmo_groups.append(dict(task=task, context=length, n=len(group),
            full_correct=correct["whole"], selected_correct=correct["teacher"],
            qc_correct=correct["cartridge"],
            full_to_qc_gained=sum(int(row["whole"]) < int(row["cartridge"]) for row, _ in group),
            full_to_qc_lost=sum(int(row["whole"]) > int(row["cartridge"]) for row, _ in group),
            median_cartridge_full_ratio=statistics.median(ratio for _, ratio in group)))
    totals = {name: sum(int(row[name]) for row in olmo)
              for name in ("whole", "teacher", "cartridge")}
    if totals != {"whole": 361, "teacher": 412, "cartridge": 446}:
        raise ValueError("OLMo accuracy totals changed")
    complete = [row for row in olmo if row["working_document"] == "complete"]
    if len(complete) != 838 or any(float(row["fact_cover_lower"]) != float(row["fact_cover_upper"])
                                   for row in complete):
        raise ValueError("OLMo fact coverage is ambiguous or the cohort changed")
    coverage_incomplete = sum(float(row["fact_cover_lower"]) < 1 for row in complete)
    if coverage_incomplete != 314:
        raise ValueError("OLMo support coverage changed")

    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected scoring version")
    refs = json.loads(args.qwen_references.read_text())
    observations = defaultdict(dict)
    with args.qwen_arms.open(newline="") as stream:
        for row in csv.DictReader(stream):
            arm, key = row["arm"], row["case"]
            if arm not in ARMS:
                continue
            if key not in refs or arm in observations[key]:
                raise ValueError("Unexpected or duplicate Qwen case")
            ref = refs[key]
            if ref["benchmark"] != "babilong" or ref["task"] != row["task"]:
                raise ValueError("Qwen reference mismatch")
            score = int(score_prediction(row["prediction"], ref["answers"], "babilong", ref["task"])[0])
            observations[key][arm] = (score, row["task"], int(row["context"]))
    qgroups = defaultdict(list)
    for key, case in observations.items():
        if set(case) != set(ARMS):
            raise ValueError("Unpaired Qwen case")
        task, length = case["vanilla"][1:]
        if any(value[1:] != (task, length) for value in case.values()):
            raise ValueError("Inconsistent Qwen task/length")
        qgroups[task, length].append(case)
    if len(observations) != 1000 or len(qgroups) != 40 or any(len(v) != 25 for v in qgroups.values()):
        raise ValueError("Expected 1,000 Qwen cases in forty task/length groups")
    qwen_groups = []
    qsummary = {}
    for arm in ARMS[1:]:
        qsummary[arm] = dict(gained=0, lost=0, correct=0)
        for (task, length), cases in sorted(qgroups.items(), key=lambda x: (x[0][1], x[0][0])):
            gained = sum(c[arm][0] > c["vanilla"][0] for c in cases)
            lost = sum(c[arm][0] < c["vanilla"][0] for c in cases)
            correct = sum(c[arm][0] for c in cases)
            qsummary[arm]["gained"] += gained
            qsummary[arm]["lost"] += lost
            qsummary[arm]["correct"] += correct
            qwen_groups.append(dict(arm=arm, task=task, context=length, n=len(cases),
                                    gained=gained, lost=lost, correct=correct))
    if qsummary != {"kv_whole_ntp": {"gained": 62, "lost": 82, "correct": 522},
                    "kv_selected_ntp": {"gained": 59, "lost": 97, "correct": 504},
                    "kv_sd_soft": {"gained": 145, "lost": 109, "correct": 578}}:
        raise ValueError("Qwen paired totals changed")
    baseline = sum(case["vanilla"][0] for case in observations.values())
    if baseline != 542:
        raise ValueError("Qwen vanilla total changed")

    context_summary = {}
    for length in LENGTHS:
        items = [item for (task, context), group in grouped.items() if context == length for item in group]
        context_summary[str(length)] = dict(n=len(items),
            full_correct=sum(int(row["whole"]) for row, _ in items),
            selected_correct=sum(int(row["teacher"]) for row, _ in items),
            qc_correct=sum(int(row["cartridge"]) for row, _ in items),
            median_cartridge_full_ratio=statistics.median(ratio for _, ratio in items))
    summary = dict(olmo=dict(n=1000, full_correct=361, selected_correct=412, qc_correct=446,
        median_cartridge_full_ratio=statistics.median(ratio for group in grouped.values() for _, ratio in group),
        fact_coverage_n=len(complete), fact_coverage_incomplete=coverage_incomplete,
        context_compacted=162, selection_and_qc_answer_format_differs=True,
        by_context=context_summary),
        qwen=dict(n=1000, vanilla_correct=baseline, methods=qsummary),
        metric=BABILONG_SCORER_VERSION, group_definition="task × context length; 25 cases",
        no_new_model_or_api_calls=True)
    write_csv(args.output / "olmo_task_length.csv", olmo_groups)
    write_csv(args.output / "qwen_objective_changes.csv", qwen_groups)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
