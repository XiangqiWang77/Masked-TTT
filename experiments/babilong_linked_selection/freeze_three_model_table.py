#!/usr/bin/env python3
"""Freeze completed 1,000-case comparisons from existing aggregate reports."""
import argparse
import json
from pathlib import Path


def collect(original, linked):
    if (original["metric"] != "answer_focused_lexical_v2" or linked["metric"] != "answer_focused_lexical_v2" or
        original["source_questions_per_model"] != 250 or linked["selected_fraction"] != .1):
        raise ValueError("The scorer, source count or selection budget changed")
    rows = {}
    for model in ("olmo", "qwen", "gpt"):
        value = original["models"][model]["overall"]
        if value["n"] != 1000:
            raise ValueError("The original model run is incomplete")
        rows[model] = dict(cases=value["n"], whole=value["whole_accuracy"],
            teacher=value["teacher_accuracy"], zero=value.get("selective_accuracy"),
            cartridge=value["cartridge_accuracy"],
            vs_teacher_pp=value["vs_teacher"]["delta_points"],
            vs_zero_pp=(value["vs_selective"]["delta_points"] if value.get("vs_selective") else None))
    modified = {}
    for model in ("qwen", "gpt"):
        entry = linked["models"][model]
        if entry["state"] != "complete" or entry["combined"]["n"] != 1000:
            raise ValueError("The linked-selection 200+800 split is incomplete")
        row = entry["combined"]
        a, pairs = row["accuracy"], row["pairs"]
        modified[model] = dict(cases=row["n"], whole=a["whole"], teacher=a["teacher"],
            zero=a["zero"], cartridge=a["cartridge"],
            vs_teacher_pp=pairs["teacher_to_cartridge"]["delta_pp"],
            vs_zero_pp=pairs["sd_original_format"]["delta_pp"],
            vs_original_selection_pp=pairs["selection_and_method"]["delta_pp"],
            teacher_format_zero=a["zero_teacher_format"],
            teacher_format_cartridge=a["cartridge_teacher_format"])
    return dict(metric=original["metric"], source_questions_per_model=250, context_lengths=4,
        original_selection=rows, linked_selection=modified,
        original_olmo_zero_update_not_saved=True,
        olmo_working_documents_compacted=162,
        teacher_has_extra_generic_answer_instruction=True,
        teacher_format_controls_before_source_release=True,
        labels_used_only_after_generation=True, new_model_calls=0, online_api_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--linked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(json.loads(args.original.read_text()), json.loads(args.linked.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
