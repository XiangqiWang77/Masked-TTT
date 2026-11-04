#!/usr/bin/env python3
"""Summarize GPT linked-value physical KV memory from its existing report."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

try:
    # The hourly hook stores both lightweight scripts in its monitor directory.
    from canonical_memory_report_20261104 import group_summary, memory_record
except ModuleNotFoundError:
    from experiments.value_lr_physical.canonical_memory_report import group_summary, memory_record

GIB = 1024 ** 3


def read_rows(root, report, allowed, expected, mode):
    if (report["mode"] != mode or report["model"] != "gpt" or report["expected"] != expected or
        report["metric"] != "answer_focused_lexical_v2" or report["fixed_value_learning_rate"] != .001 or
        report["cached_scores"] is not True or report["online_api_calls"] != 0):
        raise ValueError("The frozen GPT score report changed")
    rows, seen = [], set()
    for saved in report["rows"]:
        key = saved["case"]
        if key in seen or key not in allowed:
            raise ValueError("A saved case is duplicated or outside the protocol")
        seen.add(key)
        result = json.loads((root / "cases/gpt" / key / "result.json").read_text())
        if (result["key"] != key or result["model_short"] != "gpt" or
            result["task"] != saved["task"] or result["length"] != saved["length"] or
            result["training"]["value_learning_rate"] != .001 or
            result["training"]["objective"] != "on-policy KL(teacher || student)" or
            result["selection"]["selection_policy"] != "decision_entity_link_v1"):
            raise ValueError("The saved physical cartridge differs from the score report")
        rows.append(dict(task=saved["task"], length=saved["length"], **memory_record(result)))
    return rows, seen


def collect(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "latest.json").read_text())
    mode = manifest["mode"]
    if (mode not in ("development", "holdout") or manifest["models"] != ["gpt"] or
        manifest["case_count"] != (200 if mode == "development" else 800) or
        manifest["case_ids"] != manifest["development_case_ids" if mode == "development" else "holdout_case_ids"] or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["fixed_value_learning_rate"] != .001 or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The fixed GPT linked-value protocol changed")
    rows, seen = read_rows(root, report, set(manifest["case_ids"]), manifest["case_count"], mode)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=mode, model="gpt",
        expected=manifest["case_count"], model_calls=0, online_api_calls=0,
        reference_answers_used=False, prediction_text_used=False,
        measured_storage="document KV only; question/generated suffix excluded",
        peak_gpu_includes_model_and_training=True, **group_summary(rows))
    if mode == "holdout" and len(seen) == 800:
        if seen != set(manifest["holdout_case_ids"]) or report["errors"] or report["invalid"]:
            raise ValueError("The completed GPT holdout case set changed")
        development = json.loads((root / "development_snapshot.json").read_text())
        if development["errors"] or development["invalid"]:
            raise ValueError("The saved GPT development report is invalid")
        dev_root = root.parent / "query_cartridges_linked_value_gpt_dev_20261104"
        dev_rows, dev_seen = read_rows(dev_root, development, set(manifest["development_case_ids"]), 200, "development")
        if (len(dev_seen) != 200 or dev_seen != set(manifest["development_case_ids"]) or dev_seen & seen or
            {key.split("-", 2)[2] for key in dev_seen} & {key.split("-", 2)[2] for key in seen}):
            raise ValueError("Development and holdout are incomplete or share source questions")
        result["combined_1000"] = group_summary(dev_rows + rows)
    return result


def render(data):
    def gib(value):
        return "—" if value is None else f"{value/GIB:.2f}"
    lines = ["# GPT linked-value standalone document KV storage", "",
        "| Scope | n | Full KV (GiB) | Selected source KV (GiB) | Cartridge (GiB) | Cartridge / full | KV reduction |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def add(label, item):
        if not item["n"]:
            lines.append(f"| {label} | 0 | — | — | — | — | — |")
            return
        lines.append(f"| {label} | {item['n']} | {gib(item['median_full_document_kv_bytes'])} | "
            f"{gib(item['median_selected_source_kv_bytes'])} | {gib(item['median_standalone_cartridge_bytes'])} | "
            f"{item['median_cartridge_percent_of_full']:.2f}% | {item['median_document_kv_reduction_percent']:.2f}% |")
    add(data["mode"], data["overall"])
    if "combined_1000" in data:
        add("combined", data["combined_1000"]["overall"])
    lines += ["", "Values are medians across cases. The ratio is calculated per case before taking the median. "
        "Document KV is measured after full/source KV and training state have been released; "
        "question and generated suffix are excluded. It is not total GPU memory or training peak.",
        "The JSON also records model-resident, training-peak and pre-decode allocated GPU bytes.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.job_root)
    for suffix, text in ((".json", json.dumps(data, indent=2) + "\n"), (".md", render(data))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
