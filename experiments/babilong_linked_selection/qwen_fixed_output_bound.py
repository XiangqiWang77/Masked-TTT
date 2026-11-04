#!/usr/bin/env python3
"""Upper bound of choosing between already generated Qwen outputs."""
import argparse
import json
import math
from pathlib import Path


def correct_count(percent, n):
    count = percent * n / 100
    if not math.isfinite(count) or not math.isclose(count, round(count), abs_tol=1e-8):
        raise ValueError("Accuracy does not represent an integral number of cases")
    return int(round(count))


def analyze(snapshot):
    if (snapshot["metric"] != "answer_focused_lexical_v2" or snapshot["selected_fraction"] != .1 or
        snapshot["selection_policy"] != "decision_entity_link_v1" or snapshot["online_api_calls"] != 0 or
        snapshot["evaluation_answers_used_only_posthoc"] is not True or
        snapshot["models"]["qwen"]["state"] != "complete"):
        raise ValueError("The completed Qwen protocol changed")
    combined = snapshot["models"]["qwen"]["combined"]
    n, acc, pairs = combined["n"], combined["accuracy"], combined["pairs"]
    if n != 1000:
        raise ValueError("The bound requires the same 1,000 questions")
    teacher = correct_count(acc["teacher"], n)
    summaries = {}
    for label, zero_key, trained_key, pair_key in (
        ("original_question", "zero", "cartridge", "sd_original_format"),
        ("teacher_requirements", "zero_teacher_format", "cartridge_teacher_format", "sd_teacher_format"),
        ("teacher_vs_original_cartridge", "teacher", "cartridge", "teacher_to_cartridge"),
    ):
        zero, trained = correct_count(acc[zero_key], n), correct_count(acc[trained_key], n)
        new, lost = pairs[pair_key]["new_correct"], pairs[pair_key]["new_wrong"]
        if (type(new) is not int or type(lost) is not int or min(new, lost) < 0 or
            zero + new - lost != trained or new > n-zero or lost > zero or
            not math.isclose(pairs[pair_key]["delta_pp"], 100*(new-lost)/n, abs_tol=1e-8)):
            raise ValueError("The paired outcomes do not match the recorded accuracies")
        # A hypothetical answer-aware chooser can recover every new correct
        # answer while retaining every originally correct answer.
        oracle = zero + new
        summaries[label] = dict(zero_correct=zero, trained_correct=trained,
            new_correct=new, new_wrong=lost, oracle_correct=oracle,
            oracle_accuracy=100*oracle/n, oracle_minus_teacher_pp=100*(oracle-teacher)/n,
            gap_to_teacher_plus_three_pp=100*(teacher+0.03*n-oracle)/n)
    return dict(n=n, metric=snapshot["metric"], teacher_correct=teacher,
                teacher_accuracy=100*teacher/n, target_accuracy=100*teacher/n+3,
                summaries=summaries, bound_is_posthoc_answer_aware=True,
                applies_only_to_each_pair_of_already_generated_outputs=True,
                evaluation_answers_never_available_to_inference=True,
                teacher_format_outputs_are_pre_release_diagnostics=True,
                new_model_calls=0, online_api_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.snapshot.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
