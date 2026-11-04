#!/usr/bin/env python3
"""Freeze aggregate quality and per-request KV sizes from existing logs only."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys


ARMS = ("vanilla", "kv_whole_ntp", "kv_selected_ntp", "kv_sd_soft")


def capture(olmo_snapshot, olmo_cases, arm_csv, references_path, scorer_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    with olmo_snapshot.open(newline="") as stream:
        olmo = list(csv.DictReader(stream))
    if len(olmo) != 1000 or len({row["case"] for row in olmo}) != 1000:
        raise ValueError("Expected the complete OLMo 1,000-case snapshot")
    groups = {"0%": [], "1–99%": [], "100%": []}
    storage = []
    excluded = 0
    for i, row in enumerate(olmo):
        result = json.loads((olmo_cases / row["case"] / "masked_ttt.json").read_text())
        memory, cartridge = result["memory"], result["standalone_cartridge"]["memory"]
        full = memory["full_document_kv_bytes"]
        compact = cartridge["standalone_cartridge_bytes"]
        materialized = cartridge["materialized_selected_kv_bytes"]
        adapter = cartridge["cartridge_adapter_bytes"]
        if (result["key"] != row["case"] or not result["backbone_unchanged"] or
            not result["original_document_KV_unchanged"] or full <= 0 or materialized <= 0 or
            compact != materialized + adapter or cartridge["raw_full_document_kv_used_at_decode"] or
            cartridge["includes_generation_suffix_kv"] or
            cartridge["selected_kv_representation"] != "independently_prefilled_selected_text"):
            raise ValueError("Unexpected OLMo document-KV accounting")
        storage.append(dict(row=i, context_tokens=int(row["context_tokens"]),
            working_document=row["working_document"], full_bytes=full,
            cartridge_bytes=compact, materialized_kv_bytes=materialized, adapter_bytes=adapter))
        if row["working_document"] != "complete":
            excluded += 1
            continue
        lo, hi = float(row["fact_cover_lower"]), float(row["fact_cover_upper"])
        if lo != hi or not 0 <= lo <= 1:
            raise ValueError("Ambiguous fact coverage")
        group = "0%" if lo == 0 else "100%" if lo == 1 else "1–99%"
        groups[group].append(row)
    if excluded != 162 or sum(len(v) for v in groups.values()) != 838:
        raise ValueError("The OLMo coverage exclusions changed")
    coverage = []
    for group, values in groups.items():
        for method in ("whole", "teacher", "cartridge"):
            correct = sum(int(row[method]) for row in values)
            coverage.append(dict(group=group, method=method, n=len(values), correct=correct,
                                 accuracy_pct=100 * correct / len(values)))

    sys.path.insert(0, str(scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected BABILONG scoring version")
    refs = json.loads(references_path.read_text())
    if len(refs) != 1000:
        raise ValueError("Expected the fixed 1,000-case Qwen references")
    observations, updates = {}, {arm: Counter() for arm in ARMS}
    with arm_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            arm, key = row["arm"], row["case"]
            if arm not in ARMS:
                continue
            if (key, arm) in observations or key not in refs:
                raise ValueError("Duplicate or unexpected Qwen case/arm")
            ref = refs[key]
            if ref["task"] != row["task"] or ref["benchmark"] != "babilong":
                raise ValueError("Qwen task or benchmark differs from its reference")
            observations[key, arm] = int(score_prediction(row["prediction"], ref["answers"],
                                                           "babilong", ref["task"])[0])
            updates[arm][row["updates"]] += 1
    cases = sorted(key for key in refs if all((key, arm) in observations for arm in ARMS))
    if len(cases) != 1000:
        raise ValueError("The Qwen methods do not all have the same 1,000 cases")
    quality = []
    for arm in ARMS:
        correct = sum(observations[key, arm] for key in cases)
        gains = sum(observations[key, arm] > observations[key, "vanilla"] for key in cases)
        losses = sum(observations[key, arm] < observations[key, "vanilla"] for key in cases)
        quality.append(dict(arm=arm, n=len(cases), correct=correct, accuracy_pct=100*correct/len(cases),
                            new_correct_vs_vanilla=gains, new_wrong_vs_vanilla=losses))

    for filename, rows in (("coverage.csv", coverage), ("objective_accuracy.csv", quality),
                           ("storage_points.csv", storage)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    ratios = [row["cartridge_bytes"] / row["full_bytes"] for row in storage]
    metadata = dict(olmo_model="OLMo-3-7B-Instruct", olmo_cases=1000,
        coverage_cases=838, coverage_excluded_compacted_documents=excluded,
        coverage_bins={key: len(value) for key, value in groups.items()},
        selected_icl_is_saved_selected_only_answer=True,
        selected_icl_has_generic_answer_instruction=True,
        cartridge_uses_original_question_format=True,
        qwen_model="Qwen3-4B-Instruct-2507", qwen_paired_cases=len(cases),
        qwen_max_update_steps=16, qwen_update_counts={arm: dict(counts) for arm, counts in updates.items()},
        qwen_sd_is_fixed_step_soft_sd_not_historical_selective_sd=True,
        metric=BABILONG_SCORER_VERSION,
        median_cartridge_to_full_document_kv_ratio=statistics.median(ratios),
        storage_cartridge_includes_materialized_kv_and_adapter=True,
        storage_excludes_model_question_answer_and_training_peak=True,
        storage_not_a_measurement_of_physical_peak_gpu_memory=True,
        no_new_model_or_online_api_calls=True, reference_answers_used_only_for_posthoc_scoring=True,
        source_olmo_snapshot=str(olmo_snapshot), source_olmo_results=str(olmo_cases),
        source_qwen_outputs=str(arm_csv))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dict(coverage=coverage, quality=quality, median_storage_ratio=statistics.median(ratios))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("olmo-snapshot", "olmo-cases", "arm-csv", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.olmo_snapshot, args.olmo_cases, args.arm_csv, args.references, args.scorer_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
