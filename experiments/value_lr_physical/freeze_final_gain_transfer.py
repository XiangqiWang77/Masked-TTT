#!/usr/bin/env python3
"""Freeze the completed 40-pair readout and update decomposition as aggregates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from experiments.value_lr_physical.report_gain_transfer import collect


def freeze(root):
    manifest = json.loads((root / "manifest.json").read_text())
    result = collect(root)  # Revalidates all saved training, readout and release contracts.
    cases = manifest["case_ids"]
    by_source = defaultdict(set)
    for key in cases:
        fields = key.split("-", 2)
        if len(fields) != 3 or fields[0] != "babilong" or fields[1] not in ("8", "16", "32", "64"):
            raise ValueError("Unexpected development case identity")
        by_source[fields[2]].add(fields[1])
    if (len(cases) != 40 or len(set(cases)) != 40 or len(by_source) != 10 or
        any(x != {"8", "16", "32", "64"} for x in by_source.values())):
        raise ValueError("Expected ten source questions at four lengths")
    models = {}
    for model in ("qwen", "gpt"):
        content = result["models"][model]
        if content["overall"]["cases"] != 40 or content["overall"]["readouts"] != 40:
            raise ValueError("Both models need all 40 validated readouts")
        reads = [json.loads((root / "cases" / model / (case + ".json")).read_text())["readout"]
                 for case in cases]
        full = [row["kl_full_native_to_full_adapted"] for row in reads]
        selected = [row["kl_selected_native_to_selected_adapted"] for row in reads]
        ratios = [b / a for a, b in zip(full, selected) if a > 1e-8]
        overall = content["overall"]
        if len(ratios) != overall["effect_ratio_count"]:
            raise ValueError("Ratio denominator filter changed")
        models[model] = dict(n=40, median_full_first_token_kl=statistics.median(full),
            median_selected_first_token_kl=statistics.median(selected),
            median_selected_over_full_first_token_kl=statistics.median(ratios) if ratios else None,
            ratio_count=len(ratios), closer_to_native_full=overall["closer_to_full"],
            farther_from_native_full=overall["farther_from_full"],
            median_first_gain_gradient_common_squared_fraction=overall["median_first_gradient_common_squared_fraction"],
            median_first_gain_update_common_squared_fraction=overall["median_first_update_common_squared_fraction"],
            update_common_over_90_percent=overall["update_common_over_90_percent"],
            total_nonzero_updates=overall["nonzero_updates"])
    return dict(scope="completed 40-case development diagnostic per model", source_questions=10,
        context_lengths_per_source=4, same_original_question_first_answer_position=True,
        selected_cache_independently_prefilled=True, reference_answers_used=False, online_api_calls=0,
        extra_training_loss=False, quality_or_correctness_not_measured=True, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = freeze(args.job_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(args.output)
    for model, item in data["models"].items():
        print(model, "cases", item["n"], "ratio", item["median_selected_over_full_first_token_kl"])


if __name__ == "__main__":
    main()
