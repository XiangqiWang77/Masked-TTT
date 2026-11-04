#!/usr/bin/env python3
"""Audit the same-cache output-format control in completed experiments.

This is a post-generation, descriptive analysis. It loads no model and never
uses question identities, evidence, references, or generated answers as output.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def group(rows):
    counts = Counter()
    counts["n"] = 0
    for row in rows:
        counts["n"] += 1
        for key in ("format_tokens_differ", "format_first_token_diff",
                    "teacher_vs_zero_fmt_diff", "teacher_vs_rollout_first_diff",
                    "trained_vs_zero_tokens_differ"):
            counts[key] += bool(row[key])
        for base, candidate, prefix in (("zero", "trained", "sd"),
                                        ("zero", "zero_format", "format")):
            counts[prefix + "_new_correct"] += row[base] == 0 and row[candidate] == 1
            counts[prefix + "_new_wrong"] += row[base] == 1 and row[candidate] == 0
        for key in ("zero", "trained", "teacher", "zero_format"):
            counts[key + "_correct"] += row[key]
    n = counts["n"]
    result = dict(counts)
    if n:
        for key in ("zero", "trained", "teacher", "zero_format"):
            result[key + "_accuracy"] = 100.0 * counts[key + "_correct"] / n
        for prefix in ("sd", "format"):
            result[prefix + "_delta_pp"] = 100.0 * (
                counts[prefix + "_new_correct"] - counts[prefix + "_new_wrong"]) / n
    return result


def read_row(result, scores):
    control = result["standalone_cartridge"]["paired_prompt_control"]
    if (not control["same_selected_kv"] or not control["same_backbone_and_device"] or
        not control["same_generation_limit"] or control["control_update_steps"] != 0 or
        control["teacher_answer_in_student_prompt"] or control["reference_answers_used"]):
        raise ValueError("The paired output controls do not share the required inputs")
    if result["training"]["objective"] != "on-policy KL(teacher || student)":
        raise ValueError("Unexpected training objective")
    for key in ("zero", "trained", "teacher", "zero_format"):
        if scores[key] not in (0, 1):
            raise ValueError("Expected binary unified post-generation scores")
    original = control["zero_original"]["tokens"]
    formatted = control["zero_teacher_suffix"]["tokens"]
    trained = control["trained_original"]["tokens"]
    teacher = result["teacher"]["tokens"]
    steps = result["training"]["steps"]
    if not steps:
        raise ValueError("Missing on-policy training step")
    rollout = steps[0]["student_rollout_tokens"]
    if any(not isinstance(x, list) or not x for x in (original, formatted, trained, teacher, rollout)):
        raise ValueError("Missing token sequences")
    return dict(scores, format_tokens_differ=original != formatted,
                format_first_token_diff=original[0] != formatted[0],
                teacher_vs_zero_fmt_diff=teacher != formatted,
                teacher_vs_rollout_first_diff=teacher[0] != rollout[0],
                trained_vs_zero_tokens_differ=trained != original)


def load(workspace, model, dataset):
    if dataset == "historical":
        folder = {"qwen": "query_cartridges_qwen_olmo1000_20261104",
                  "gpt": "query_cartridges_gpt_olmo1000_20261103"}[model]
        root = workspace / folder
        report = json.loads((root / "latest.json").read_text())
        if report["completed"] != 1000 or report["metric"] != "answer_focused_lexical_v2":
            raise ValueError("Expected a complete 1,000-case report")
        records, expected = report["rows"], 1000
        names = dict(zero="selective", trained="cartridge", teacher="teacher",
                     zero_format="selective_teacher_format")
        path_for = lambda case: root / "cases" / case / "weight_0.00.json"
    else:
        root = workspace / "query_cartridges_linked_selection_20261104"
        report = json.loads((root / "latest.json").read_text())
        if report["metric"] != "answer_focused_lexical_v2":
            raise ValueError("Expected the same task-aware scorer")
        records, expected = report["models"][model]["rows"], 200
        names = dict(zero="zero", trained="cartridge", teacher="teacher",
                     zero_format="zero_teacher_format")
        path_for = lambda case: root / "cases" / model / case / "result.json"
    if len(records) != expected or len({item["case"] for item in records}) != expected:
        raise ValueError("Wrong number of complete, unique examples")
    result = []
    for item in records:
        document = json.loads(path_for(item["case"]).read_text())
        row = read_row(document, {key: item[value] for key, value in names.items()})
        row["task"] = item["task"]
        result.append(row)
    return result


def summarize(rows):
    summary = dict(overall=group(rows), by_original_format_first_token_difference={
        label: group([row for row in rows if row["format_first_token_diff"] == flag])
        for label, flag in (("same", False), ("different", True))})
    if rows and all("task" in row for row in rows):
        summary["by_task"] = {task: group([row for row in rows if row["task"] == task])
                              for task in sorted({row["task"] for row in rows})}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = dict(metric="answer_focused_lexical_v2", post_generation_only=True,
                new_model_calls=0, new_api_calls=0,
                note="The two output-format groups are descriptive, not causal.",
                datasets={dataset: {model: summarize(load(args.workspace, model, dataset))
                                   for model in ("qwen", "gpt")}
                          for dataset in ("historical", "linked_dev")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for dataset, models in data["datasets"].items():
        for model, summary in models.items():
            print(dataset, model, json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
