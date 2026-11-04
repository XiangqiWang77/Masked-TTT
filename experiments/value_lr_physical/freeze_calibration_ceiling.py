#!/usr/bin/env python3
"""Upper-bound fixed-output calibration on the completed development cohort."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

FAMILIES = {"value_learning_rate": ("lr_0.0001", "lr_0.0003", "lr_0.0010"),
            "update_steps": ("step2", "step4", "step8")}


def summarize(rows, arms):
    n = len(rows)
    counts = {name: sum(row[name] for row in rows) for name in (*arms, "teacher", "whole", "zero")}
    oracle = sum(any(row[name] for name in arms) for row in rows)
    varying = sum(len({row[name] for name in arms}) > 1 for row in rows)
    return dict(n=n, accuracy={name: 100 * value / n if n else None for name, value in counts.items()},
        oracle_accuracy=100 * oracle / n if n else None,
        oracle_above_teacher_pp=100 * (oracle - counts["teacher"]) / n if n else None,
        oracle_above_whole_pp=100 * (oracle - counts["whole"]) / n if n else None,
        oracle_correct=oracle, answer_changes_across_candidates=varying,
        all_candidates_wrong=n - oracle)


def freeze(value_manifest, value_report, step_manifest, step_report):
    keys = value_manifest["case_ids"]
    if (len(keys) != 200 or len(set(keys)) != 200 or set(step_manifest["case_ids"]) != set(keys) or
        value_manifest["model_protocols"] != step_manifest["model_protocols"] or
        value_manifest["keep_fraction"] != .1 or step_manifest["keep_fraction"] != .1 or
        value_manifest["basis_source"] != "document" or step_manifest["basis_source"] != "document" or
        value_manifest["selected_transfer_weight"] != 0 or step_manifest["selected_transfer_weight"] != 0 or
        value_manifest["reference_answers_used"] is not False or
        step_manifest["reference_answers_used_in_inference"] is not False or
        value_manifest["value_learning_rates"] != [.0001, .0003, .001] or
        step_manifest["step_budgets"] != [2, 4, 8]):
        raise ValueError("The fixed OLMo-style calibration protocols changed")
    sources = defaultdict(set)
    for key in keys:
        fields = key.split("-", 2)
        if len(fields) != 3 or fields[0] != "babilong" or fields[1] not in ("8", "16", "32", "64"):
            raise ValueError("Unexpected development case identity")
        sources[fields[2]].add(fields[1])
    if len(sources) != 50 or any(lengths != {"8", "16", "32", "64"} for lengths in sources.values()):
        raise ValueError("Expected 50 source questions with four lengths")
    for report in (value_report, step_report):
        if (report["metric"] != "answer_focused_lexical_v2" or report["development_only"] is not True or
            report["online_api_calls"] != 0 or report["errors"] or report["invalid"]):
            raise ValueError("The completed calibration report changed")
    if (value_report["same_teacher_student_kl"] is not True or value_report["same_document_basis"] is not True or
        value_report["evaluation_answers_used_in_training"] is not False or
        step_report["same_trajectory"] is not True or step_report["evaluation_answers_used_only_posthoc"] is not True):
        raise ValueError("The training or evaluation contract changed")
    models = {}
    for model in ("qwen", "gpt"):
        families = {}
        for label, report in (("value_learning_rate", value_report), ("update_steps", step_report)):
            rows = report["models"][model]["rows"]
            arms = FAMILIES[label]
            if len(rows) != 200 or {row["case"] for row in rows} != set(keys):
                raise ValueError("The paired candidate outputs are incomplete")
            if any(type(row.get(name)) is not int or row[name] not in (0, 1)
                   for row in rows for name in (*arms, "teacher", "whole", "zero")):
                raise ValueError("Expected binary, complete post-generation scores")
            grouped = defaultdict(list)
            for row in rows:
                grouped["task:" + row["task"]].append(row)
                grouped["length:" + str(row["length"])].append(row)
            families[label] = dict(overall=summarize(rows, arms),
                groups={name: summarize(items, arms) for name, items in sorted(grouped.items())})
        models[model] = families
    return dict(scope="200-case development cohort; 50 sources at four lengths", metric=value_report["metric"],
        oracle_definition="post-hoc gold-aware choice among recorded fixed candidate outputs; not deployable",
        reference_answers_used_in_inference=False, reference_answers_used_only_after_generation=True,
        online_api_calls=0, new_model_calls=0, candidate_families=FAMILIES, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("value-manifest", "value-report", "step-manifest", "step-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    data = freeze(json.loads(args.value_manifest.read_text()), json.loads(args.value_report.read_text()),
                  json.loads(args.step_manifest.read_text()), json.loads(args.step_report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for model, families in data["models"].items():
        for name, info in families.items():
            print(model, name, json.dumps(info["overall"]))


if __name__ == "__main__":
    main()
