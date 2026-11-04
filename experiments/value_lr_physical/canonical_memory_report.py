#!/usr/bin/env python3
"""Summarize physically released cartridge storage from completed paired cases.

Read only scores already accepted by the standard report and the corresponding
saved result metadata. No model, API, reference answer, prediction or source
text is used, and the output contains aggregate statistics only.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

GIB = 1024 ** 3


def positive_bytes(value):
    if type(value) is not int or value < 1:
        raise ValueError("A positive measured byte count is required")
    return value


def memory_record(result):
    request = result["memory"]
    memory = result["standalone_cartridge"]["memory"]
    if (request["cache_sizes_exclude_question_and_generated_suffix"] is not True or
        request["gpu_peak_includes_model_and_training"] is not True or
        request["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0 or
        not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
        memory["raw_full_document_kv_used_at_decode"] or
        memory["selected_kv_representation"] != "independently_prefilled_selected_text" or
        memory["additional_source_kv_bytes_before_decode"] != 0 or
        memory["additional_training_adapter_bytes_before_decode"] != 0):
        raise ValueError("The independent cartridge was not physically released")
    full = positive_bytes(request["full_document_kv_bytes"])
    cartridge = positive_bytes(memory["standalone_cartridge_bytes"])
    source = positive_bytes(memory["source_selected_kv_bytes"])
    if memory["live_document_state_before_decode_bytes"] != cartridge:
        raise ValueError("The live document state includes storage outside the cartridge")
    optional = (request["model_resident_gpu_bytes"], request["paired_run_peak_gpu_allocated_bytes"],
                memory["cuda_allocated_before_decode_bytes"])
    if any(value is not None and (type(value) is not int or value < 0) for value in optional):
        raise ValueError("Invalid CUDA allocated-byte statistic")
    return dict(full=full, cartridge=cartridge, selected_source=source,
                model_gpu=optional[0], training_peak_gpu=optional[1], decode_gpu=optional[2])


def summarize(rows):
    n = len(rows)
    if not n:
        return dict(n=0)
    for row in rows:
        for key in ("full", "cartridge", "selected_source"):
            positive_bytes(row[key])
    def median(key):
        values = [row[key] for row in rows if row[key] is not None]
        if values and len(values) != n:
            raise ValueError("Partial CUDA accounting is not comparable")
        return statistics.median(values) if values else None
    median_ratio = statistics.median(100 * row["cartridge"] / row["full"] for row in rows)
    return dict(n=n, median_full_document_kv_bytes=median("full"),
        median_selected_source_kv_bytes=median("selected_source"),
        median_standalone_cartridge_bytes=median("cartridge"),
        median_cartridge_percent_of_full=median_ratio, median_document_kv_reduction_percent=100 - median_ratio,
        median_model_resident_gpu_bytes=median("model_gpu"),
        median_training_peak_gpu_allocated_bytes=median("training_peak_gpu"),
        median_allocated_gpu_before_decode_bytes=median("decode_gpu"))


def read_rows(root, model, saved_rows, allowed):
    rows, seen = [], set()
    for saved in saved_rows:
        key = saved["case"]
        if key in seen or key not in allowed:
            raise ValueError("A saved case is duplicated or outside the protocol")
        seen.add(key)
        result = json.loads((root / "cases" / model / key / "result.json").read_text())
        if (result["key"] != key or result["model_short"] != model or
            result["task"] != saved["task"] or result["length"] != saved["length"]):
            raise ValueError("The saved physical cartridge differs from the paired report")
        rows.append(dict(task=saved["task"], length=saved["length"], **memory_record(result)))
    return rows, seen


def group_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    return dict(overall=summarize(rows), groups={key: summarize(group) for key, group in sorted(groups.items())})


def collect(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "latest.json").read_text())
    if (report["metric"] != "answer_focused_lexical_v2" or report["mode"] != manifest["mode"] or
        report["expected_per_model"] != manifest["case_count"] or
        set(report["models"]) != set(manifest["models"]) or
        report["cached_scores"] is not True or report["online_api_calls"] != 0):
        raise ValueError("The paired evaluation report and manifest differ")
    models = {}
    for model in manifest["models"]:
        rows, seen = read_rows(root, model, report["models"][model]["rows"], set(manifest["case_ids"]))
        models[model] = group_summary(rows)
        if manifest["mode"] == "holdout" and len(seen) == 800:
            if (seen != set(manifest["holdout_case_ids"]) or
                any(item["model"] == model for item in report["errors"] + report["invalid"])):
                raise ValueError("The completed holdout case set changed")
            development = json.loads((root / "development_snapshot.json").read_text())
            if (development["metric"] != report["metric"] or development["mode"] != "development" or
                development["expected_per_model"] != 200 or development["online_api_calls"] != 0 or
                model not in development["models"] or any(item["model"] == model for item in
                    development["errors"] + development["invalid"])):
                raise ValueError("The saved development report changed")
            dev_root = root.parent / "query_cartridges_canonical_pair_dev_20261104"
            dev_rows, dev_seen = read_rows(dev_root, model, development["models"][model]["rows"],
                                           set(manifest["development_case_ids"]))
            if (len(dev_seen) != 200 or dev_seen != set(manifest["development_case_ids"]) or
                dev_seen & seen or {key.split("-", 2)[2] for key in dev_seen} &
                {key.split("-", 2)[2] for key in seen}):
                raise ValueError("Development and holdout are incomplete or share source questions")
            models[model]["combined_1000"] = group_summary(dev_rows + rows)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=manifest["mode"],
        expected_per_model=manifest["case_count"], metric=report["metric"], model_calls=0,
        online_api_calls=0, reference_answers_used=False, prediction_text_used=False,
        measured_storage="document KV only; question/generated suffix excluded",
        peak_gpu_includes_model_and_training=True, models=models)


def render(data):
    def gib(value):
        return "—" if value is None else f"{value/GIB:.2f}"
    lines = ["# Standalone cartridge document KV storage", "",
        "| Model | n | Full KV (GiB) | Selected source KV (GiB) | Cartridge (GiB) | Cartridge / full | KV reduction |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def add_row(label, s):
        if not s["n"]:
            lines.append(f"| {label} | 0 | — | — | — | — | — |")
            return
        lines.append(f"| {label} | {s['n']} | {gib(s['median_full_document_kv_bytes'])} | "
            f"{gib(s['median_selected_source_kv_bytes'])} | {gib(s['median_standalone_cartridge_bytes'])} | "
            f"{s['median_cartridge_percent_of_full']:.2f}% | {s['median_document_kv_reduction_percent']:.2f}% |")
    for model, entry in data["models"].items():
        add_row(model + " " + data["mode"], entry["overall"])
        if "combined_1000" in entry:
            add_row(model + " combined", entry["combined_1000"]["overall"])
    lines += ["", "Values are medians across cases. The ratio is computed per case before taking the median. "
        "This is physical document KV storage after releasing both source caches and the training state; "
        "it excludes the question and generated suffix and is not total GPU memory or training peak.",
        "The aggregate JSON separately records model-resident, training-peak and pre-decode allocated GPU "
        "bytes where available; the training peak includes model weights and training state.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.candidate_root)
    for suffix, text in ((".json", json.dumps(result, indent=2) + "\n"), (".md", render(result))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
