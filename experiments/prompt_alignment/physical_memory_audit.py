#!/usr/bin/env python3
"""Aggregate physical cache sizes from completed aligned-question development."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from experiments.value_lr_physical.canonical_memory_report import memory_record, summarize


def collect(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "latest.json").read_text())
    done = json.loads((root / "finished.json").read_text())
    keys = manifest["case_ids"]
    if (manifest["mode"] != "development" or manifest["case_count"] != 200 or
        len(keys) != 200 or len(set(keys)) != 200 or manifest["models"] != ["qwen", "gpt"] or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        report["metric"] != "answer_focused_lexical_v2" or report["expected_per_model"] != 200 or
        report["errors"] or report["invalid"] or not done["all_complete"]):
        raise ValueError("Expected the complete frozen 200-case development run")
    models = {}
    for model in manifest["models"]:
        item = report["models"][model]
        saved = item["candidate_rows"]
        if (item["candidate_completed"] != 200 or len(saved) != 200 or
            {row["case"] for row in saved} != set(keys) or done["completed"][model] != 200):
            raise ValueError("The completed model does not cover the exact case set")
        rows, groups = [], defaultdict(list)
        for row in saved:
            result = json.loads((root / "cases" / model / row["case"] / "result.json").read_text())
            train, selection = result["training"], result["selection"]
            if (result["key"] != row["case"] or result["model_short"] != model or
                result["task"] != row["task"] or result["length"] != row["length"] or
                train["objective"] != "on-policy KL(teacher || student)" or
                train["student_question_matches_teacher"] is not True or train["selected_transfer_weight"] != 0 or
                selection["selection_policy"] != "decision_entity_link_v1" or
                selection["reference_answers_used"] or selection["online_api_calls"] != 0):
                raise ValueError("The completed SD or selected-evidence contract changed")
            record = memory_record(result)
            rows.append(record)
            groups["length:" + str(row["length"])].append(record)
        models[model] = dict(overall=summarize(rows),
                             groups={key: summarize(group) for key, group in sorted(groups.items())})
    return dict(scope="Completed aligned-question development; 50 source questions at four lengths per model",
        metric=report["metric"], model_calls=0, online_api_calls=0, reference_answers_used=False,
        prediction_text_used=False, peak_gpu_includes_model_and_training=True, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.job_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for name, item in data["models"].items():
        result = item["overall"]
        print(name, result["n"], "cartridge/full percent", round(result["median_cartridge_percent_of_full"], 2))


if __name__ == "__main__":
    main()
