#!/usr/bin/env python3
"""Summarize gain decompositions and readouts without publishing answers."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


def median(values):
    return statistics.median(values) if values else None


def summarize(rows):
    steps = [step for row in rows for step in row["observations"] if "unavailable" not in step]
    first = [row["observations"][0] for row in rows
             if row["observations"] and "unavailable" not in row["observations"][0]]
    gradient = [step["gradient"]["common_squared_fraction"] for step in steps]
    update = [step["optimizer_update"]["common_squared_fraction"] for step in steps]
    first_gradient = [step["gradient"]["common_squared_fraction"] for step in first]
    first_update = [step["optimizer_update"]["common_squared_fraction"] for step in first]
    gradient = [x for x in gradient if x is not None]
    update = [x for x in update if x is not None]
    first_gradient = [x for x in first_gradient if x is not None]
    first_update = [x for x in first_update if x is not None]
    readout = [row["readout"] for row in rows if "unavailable" not in row["readout"]]
    alignment = [row["value_alignment"] for row in rows
                 if row.get("value_alignment") and "unavailable" not in row["value_alignment"]]
    def alignment_median(name):
        values = [row[name] for row in alignment if row.get(name) is not None]
        return median(values)
    ratios = [r["kl_selected_native_to_selected_adapted"] / r["kl_full_native_to_full_adapted"]
              for r in readout if r["kl_full_native_to_full_adapted"] > 1e-8]
    return dict(cases=len(rows), steps=len(steps), first_steps=len(first), readouts=len(readout),
        median_gradient_common_squared_fraction=median(gradient),
        median_update_common_squared_fraction=median(update),
        median_first_gradient_common_squared_fraction=median(first_gradient),
        median_first_update_common_squared_fraction=median(first_update),
        update_common_over_90_percent=sum(value > .9 for value in update),
        nonzero_updates=len(update),
        median_selected_over_full_first_token_change=median(ratios), effect_ratio_count=len(ratios),
        value_alignment_cases=len(alignment),
        median_value_native_cosine=alignment_median("median_layer_native_cosine"),
        median_value_basis_coordinate_cosine=alignment_median("median_layer_basis_coordinate_cosine"),
        median_value_written_delta_cosine=alignment_median("median_layer_written_delta_cosine"),
        closer_to_full=sum(r["kl_full_native_to_selected_adapted"] < r["kl_full_native_to_selected_native"] for r in readout),
        farther_from_full=sum(r["kl_full_native_to_selected_adapted"] > r["kl_full_native_to_selected_native"] for r in readout))


def collect(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (len(manifest["case_ids"]) != 40 or manifest["models"] != ["qwen", "gpt"] or
        manifest["reference_answers_used"] is not False or manifest["online_api_calls"] != 0 or
        manifest["selected_transfer_weight"] != 0 or manifest["diagnosis_only"] is not True):
        raise ValueError("The frozen diagnostic protocol changed")
    models = {}
    for model in manifest["models"]:
        rows, unavailable = [], Counter()
        for key in manifest["case_ids"]:
            path = root / "cases" / model / (key + ".json")
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            training, checks, memory = row["training"], row["checks"], row["memory"]
            if (row["key"] != key or row["model"] != model or
                training["objective"] != "on-policy KL(teacher || student)" or
                training["selected_transfer_weight"] != 0 or
                row["selection"]["source_case_key"] != key or
                row["selection"]["reference_answers_used"] is not False or
                row["selection"]["online_api_calls"] != 0 or
                checks != dict(backbone_unchanged=True, original_document_KV_unchanged=True,
                               full_cache_released=True, teacher_answer_in_final_prompt=False) or
                not memory["kv_only_materialized"] or memory["additional_source_kv_bytes_before_decode"] != 0 or
                memory["additional_training_adapter_bytes_before_decode"] != 0 or
                len(row["observations"]) != len(training["steps"])):
                raise ValueError("A diagnostic result violates its frozen contract")
            for training_step, step in zip(training["steps"], row["observations"]):
                if "unavailable" in step:
                    unavailable["gain:" + step["unavailable"]] += 1
                    continue
                for name in ("gradient", "optimizer_update", "final_gain"):
                    x = step[name]
                    if (x["selected_chunks"] < 1 or any(not math.isfinite(x[k]) or x[k] < 0
                        for k in ("common_norm", "contrast_norm", "total_norm")) or
                        not math.isclose(x["common_norm"] ** 2 + x["contrast_norm"] ** 2,
                                         x["total_norm"] ** 2, rel_tol=1e-8, abs_tol=1e-15) or
                        (x["common_squared_fraction"] is not None and not 0 <= x["common_squared_fraction"] <= 1)):
                        raise ValueError("Invalid gain decomposition")
                if not math.isclose(training_step["gain_gradient_norm"], step["gradient"]["total_norm"],
                                    rel_tol=1e-5, abs_tol=1e-8):
                    raise ValueError("Recorded gradient disagrees with training")
            if "unavailable" in row["readout"]:
                unavailable["readout:" + row["readout"]["unavailable"]] += 1
            else:
                readout = row["readout"]
                if (readout["reference_answers_used"] or readout["teacher_answer_in_student_prompt"] or
                    readout["full_cache_or_model_modified"] or
                    any(not math.isfinite(readout[k]) or readout[k] < 0 for k in (
                        "kl_full_native_to_selected_native", "kl_full_native_to_selected_adapted",
                        "kl_full_native_to_full_adapted", "kl_selected_native_to_selected_adapted"))):
                    raise ValueError("Invalid first-token readout")
            alignment = row.get("value_alignment")
            if alignment and "unavailable" in alignment:
                if alignment["unavailable"] != "not_requested":
                    unavailable["value_alignment:" + alignment["unavailable"]] += 1
            elif alignment:
                if (alignment.get("reference_answers_used") is not False or
                    alignment.get("model_or_cache_modified") is not False or
                    not 0 < alignment.get("sampled_tokens", 0) <= alignment.get("selected_tokens", 0) or
                    not alignment.get("layers")):
                    raise ValueError("Invalid Value alignment diagnostic")
                for key in ("median_layer_native_cosine", "median_layer_basis_coordinate_cosine",
                            "median_layer_written_delta_cosine"):
                    value = alignment.get(key)
                    if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
                        raise ValueError("Invalid Value alignment cosine")
            rows.append(row)
        groups = defaultdict(list)
        for row in rows:
            groups[row["task"]].append(row)
        models[model] = dict(overall=summarize(rows),
            by_task={task: summarize(items) for task, items in sorted(groups.items())},
            unavailable=dict(unavailable))
    return dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        scope=manifest["scope"], expected_per_model=40, reference_answers_used=False,
        online_api_calls=0, no_extra_training_loss=True, models=models)


def write_report(root):
    result = collect(root)
    temporary = root / "latest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(root / "latest.json")
    fmt = lambda value: "—" if value is None else f"{value:.4f}"
    lines = ["# Original SD: gain transfer diagnostic", "",
        f"Updated {result['updated_utc']}; up to 40 paired development cases per model.", "",
        "| Model | Cases | First-step common gradient | First-step common update | First-token selected/full ratio |",
        "|---|---:|---:|---:|---:|"]
    for model, data in result["models"].items():
        overall = data["overall"]
        lines.append(f"| {model} | {overall['cases']} | {fmt(overall['median_first_gradient_common_squared_fraction'])} | "
                     f"{fmt(overall['median_first_update_common_squared_fraction'])} | "
                     f"{fmt(overall['median_selected_over_full_first_token_change'])} "
                     f"({overall['effect_ratio_count']}) |")
    lines += ["", "Common fractions are squared-norm fractions across selected chunks, within each layer/head. "
              "The first-token ratio compares two view-specific KL changes and is not a correctness or information-transfer score. "
              "Missing readouts and incomplete cases are separate from wrong answers. No references or API are used here."]
    temporary = root / "report.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "report.md")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    args = parser.parse_args()
    result = write_report(args.job_root)
    for model, data in result["models"].items():
        print(model, data["overall"]["cases"])


if __name__ == "__main__":
    main()
