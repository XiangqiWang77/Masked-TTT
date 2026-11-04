#!/usr/bin/env python3
"""Audit paired Value learning rates on already completed development cases.

The analysis validates the input and output pairing, then exports aggregates
only. It runs no model, reads no reference answers, and makes no API calls.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


RATES = ("lr_0.0001", "lr_0.0003", "lr_0.0010")


def validate_pair(results, case):
    if len(results) != len(RATES):
        raise ValueError("Three candidate rates are required")
    first = results[0]
    identity = lambda result: tuple(result[key] for key in ("job", "gpu", "node", "allocation_restart_count"))
    selection = lambda result: (result["selection"]["selected_original_chunks"],
                                result["selection"]["source_case_key"])
    control = lambda result: result["standalone_cartridge"]["paired_prompt_control"]
    for name, result in zip(RATES, results):
        training, chosen, paired = result["training"], result["selection"], control(result)
        if (result["key"] != case or result["value_learning_rate"] != float(name[3:]) or
            result["gain_learning_rate"] != .01 or
            training["objective"] != "on-policy KL(teacher || student)" or
            training["selected_transfer_weight"] != 0 or training["value_basis_source"] != "document" or
            not training["teacher_conditioned_on_student_prefix"] or
            not training["student_rollouts_refreshed_each_step"] or not training["steps"] or
            chosen["reference_answers_used"] or chosen["online_api_calls"] != 0 or
            chosen["source_case_key"] != case or not chosen["complete_original_document"] or
            not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
            result["self_study"]["reference_answers_used"] or result["self_study"]["student_answer_hint_used"] or
            paired["reference_answers_used"] or paired["teacher_answer_in_student_prompt"] or
            not paired["same_selected_kv"] or not paired["same_backbone_and_device"] or
            not paired["same_generation_limit"] or paired["control_update_steps"] != 0):
            raise ValueError("The original frozen-model SD contract changed")
        if (identity(result) != identity(first) or selection(result) != selection(first) or
            result["teacher"]["tokens"] != first["teacher"]["tokens"] or
            result["self_study"]["checks"] != first["self_study"]["checks"] or
            training["steps"][0]["student_rollout_tokens"] != first["training"]["steps"][0]["student_rollout_tokens"] or
            paired["zero_original"]["tokens"] != control(first)["zero_original"]["tokens"] or
            paired["zero_teacher_suffix"]["tokens"] != control(first)["zero_teacher_suffix"]["tokens"]):
            raise ValueError("A matched input, Teacher, initial prefix, or zero-update control changed")


def summarize(rows):
    n = len(rows)
    output = {}
    for rate in RATES:
        comparisons = [(row["zero"], row[rate]) for row in rows]
        if any(a not in (0, 1) or b not in (0, 1) for a, b in comparisons):
            raise ValueError("Post-generation scores must be binary")
        correct = sum(a == 0 and b == 1 for a, b in comparisons)
        wrong = sum(a == 1 and b == 0 for a, b in comparisons)
        output[rate] = dict(n=n, accuracy=100 * sum(b for _, b in comparisons) / n if n else None,
            changed_answer_sequences=sum(row[rate + "_answer_changed"] for row in rows),
            new_correct=correct, new_wrong=wrong,
            delta_vs_zero_pp=100 * (correct - wrong) / n if n else None,
            median_relative_value_change=statistics.median(row[rate + "_relative_l2"] for row in rows) if n else None,
            median_value_elements_changed=statistics.median(row[rate + "_changed"] for row in rows) if n else None)
    return output


def collect(root):
    report = json.loads((root / "latest.json").read_text())
    if (report["metric"] != "answer_focused_lexical_v2" or report["expected"] != 200 or
        not report["development_only"] or not report["same_teacher_student_kl"] or
        not report["same_document_basis"] or report["selected_transfer_weight"] != 0 or
        report["evaluation_answers_used_in_training"] or report["online_api_calls"] != 0 or
        report["errors"] or report["invalid"]):
        raise ValueError("Expected the completed original-SD Value-rate development report")
    models = {}
    for model in ("qwen", "gpt"):
        rows = report["models"][model]["rows"]
        if len(rows) != 200 or len({row["case"] for row in rows}) != 200:
            raise ValueError("Expected 200 unique completed cases per model")
        processed, groups = [], defaultdict(list)
        for row in rows:
            folder = root / "cases" / model / row["case"]
            results = [json.loads((folder / (name + ".json")).read_text()) for name in RATES]
            validate_pair(results, row["case"])
            item = dict(zero=row["zero"])
            for rate, result in zip(RATES, results):
                paired = result["standalone_cartridge"]["paired_prompt_control"]
                item[rate] = row[rate]
                item[rate + "_answer_changed"] = paired["zero_original"]["tokens"] != paired["trained_original"]["tokens"]
                item[rate + "_relative_l2"] = row[rate + "_relative_l2"]
                item[rate + "_changed"] = row[rate + "_changed"]
            processed.append(item)
            groups["task:" + row["task"]].append(item)
            groups["length:" + str(row["length"])].append(item)
        models[model] = dict(teacher_accuracy=report["models"][model]["overall"]["accuracy"]["teacher"],
            zero_accuracy=report["models"][model]["overall"]["accuracy"]["zero"],
            overall=summarize(processed), groups={key: summarize(value) for key, value in groups.items()})
    return dict(scope="200 paired development cases per model, 50 sources at four lengths",
        metric=report["metric"], same_allocation_selection_teacher_practice_initial_prefix_and_zero=True,
        selected_transfer_weight=0, model_or_api_calls=0, output_contains_case_identifiers=False,
        original_question_format=True, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.job_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for model, row in data["models"].items():
        print(model, "teacher", row["teacher_accuracy"], "zero", row["zero_accuracy"],
              json.dumps(row["overall"]))


if __name__ == "__main__":
    main()
