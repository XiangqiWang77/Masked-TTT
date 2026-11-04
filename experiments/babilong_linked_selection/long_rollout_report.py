#!/usr/bin/env python3
"""Score completed linked/32-token results after generation on fixed cases."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import (
    development_decision, holdout_decision, summarize, validate_linked_baseline, validate_partition,
)


EXPECTED_MODELS = {"qwen": "Qwen/Qwen3-4B-Instruct-2507", "gpt": "openai/gpt-oss-20b"}
INSPECTED = {"qa1-300023", "qa2-300037"}


def validate_new(case, result, model, task):
    selection, training = result["selection"], result["training"]
    final = result["standalone_cartridge"]
    controls, memory = final["original_question_controls"], final["memory"]
    release = result["memory"]["full_cache_release_before_standalone"]
    expected_chunks = max(1, math.ceil(selection["total_chunks"] * .1))
    if (result["key"] != case or result["model"] != EXPECTED_MODELS[model] or result["task"] != task or
        result["keep_fraction"] != .1 or selection["source_case_key"] != case or
        selection["backend"] != "precomputed_source_JSON" or
        selection["selection_policy"] != "decision_entity_link_v1" or
        selection["entity_reserve_fraction"] != .25 or
        len(selection["selected_chunks"]) != expected_chunks or not selection["complete_original_document"] or
        selection["reference_answers_used"] or selection["online_api_calls"] != 0 or
        training["objective"] != "on-policy KL(teacher || student)" or
        training["value_basis_source"] != "document" or training["selected_transfer_weight"] != 0 or
        training["student_rollout_limit"] != 32 or not training["steps"] or
        not training["teacher_conditioned_on_student_prefix"] or
        not training["student_rollouts_refreshed_each_step"] or
        any("selected_transfer_KL_before" in step for step in training["steps"]) or
        not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
        not result["teacher"]["response_instruction_applied"] or
        result["self_study"]["reference_answers_used"] or result["self_study"]["student_answer_hint_used"] or
        result["teacher_answer_in_final_prompt"] or final["teacher_answer_in_final_prompt"] or
        controls["reference_answers_used"] or controls["teacher_answer_in_student_prompt"] or
        not controls["same_selected_kv"] or not controls["same_question_suffix"] or
        not controls["same_generation_limit"] or controls["teacher_suffix_matches_student"] or
        controls["adapted"]["tokens"] != final["generated_token_ids"] or
        not release["requested"] or release["full_cache_tensors_live_before_standalone"] != 0 or
        not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
        memory["additional_source_kv_bytes_before_decode"] != 0 or
        memory["additional_training_adapter_bytes_before_decode"] != 0):
        raise ValueError("Combined original-method or physically independent cartridge contract failed")


def collect(manifest, candidate_root, baseline_roots, references, score):
    cases = manifest["case_ids"]
    count, mode, models = manifest["case_count"], manifest["mode"], manifest["models"]
    development, holdout = manifest["development_case_ids"], manifest["holdout_case_ids"]
    validate_partition(development, holdout)
    if (mode not in ("development", "holdout") or count not in (200, 800) or
        len(cases) != count or len(set(cases)) != count or
        (mode == "development") != (count == 200) or not models or
        len(set(models)) != len(models) or any(model not in EXPECTED_MODELS for model in models) or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["entity_reserve_fraction"] != .25 or manifest["keep_fraction"] != .1 or
        manifest["student_rollout_tokens"] != 32 or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        set(manifest["qualitative_case_sources"]) != INSPECTED or
        set(cases) != set(development if mode == "development" else holdout)):
        raise ValueError("The frozen combined evaluation protocol changed")

    all_models, errors, invalid = {}, [], []
    for model in models:
        rows = []
        for case in cases:
            folder = Path(candidate_root) / "cases" / model / case
            path = folder / "result.json"
            if not path.is_file():
                error_path = folder / "error.json"
                if error_path.is_file():
                    error = json.loads(error_path.read_text())
                    errors.append({"model": model, "case": case, "type": error.get("type")})
                continue
            result = json.loads(path.read_text())
            ref = references[case]
            try:
                validate_new(case, result, model, ref["task"])
            except (KeyError, TypeError, ValueError) as exc:
                invalid.append({"model": model, "case": case, "reason": str(exc)})
                continue
            controls = result["standalone_cartridge"]["original_question_controls"]
            grade = lambda prediction: int(score(prediction, ref["answers"], "babilong", ref["task"])[0])
            full_bytes = result["memory"]["full_document_kv_bytes"]
            artifact_bytes = result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]
            if not full_bytes or not artifact_bytes:
                raise ValueError("Missing cache accounting")
            row = {"case": case, "task": result["task"], "length": result["length"],
                   "whole": grade(result["self_judge"]["native_prediction"]),
                   "zero": grade(controls["zero"]["prediction"]),
                   "teacher": grade(result["teacher"]["prediction"]),
                   "cartridge": grade(result["prediction"]),
                   "cartridge_percent": 100 * artifact_bytes / full_bytes,
                   "linked8": None, "baseline_source": None}
            rejected = []
            for root in baseline_roots:
                old_path = Path(root) / "cases" / model / case / "result.json"
                if not old_path.is_file():
                    continue
                try:
                    old = json.loads(old_path.read_text())
                    validate_linked_baseline(case, old, result)
                    row["linked8"] = grade(old["standalone_cartridge"]["paired_prompt_control"]
                                           ["trained_original"]["prediction"])
                    row["baseline_source"] = Path(root).name
                    break
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    rejected.append(str(exc))
            if rejected and row["baseline_source"] is None:
                row["baseline_rejected"] = rejected
            rows.append(row)
        grouped = defaultdict(list)
        for row in rows:
            grouped["task:" + row["task"]].append(row)
            grouped["length:" + str(row["length"])].append(row)
        overall = summarize(rows)
        item = {"overall": overall, "groups": {key: summarize(value) for key, value in grouped.items()},
                "rows": rows}
        if mode == "development":
            item["decision"] = development_decision(overall)
        else:
            strict = [row for row in rows if row["case"].split("-", 2)[2] not in INSPECTED]
            item["excluding_viewed_examples"] = summarize(strict)
            item["decision"] = holdout_decision(item["excluding_viewed_examples"])
        if any(entry["model"] == model for entry in errors + invalid):
            item["decision"] = {"eligible": False, "reason": "result_errors_or_invalid"}
        all_models[model] = item
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "metric": "answer_focused_lexical_v2", "expected_per_model": count,
            "selection_policy": "decision_entity_link_v1", "cached_scores": True,
            "online_api_calls": 0, "evaluation_answers_used_only_posthoc": True,
            "selected_transfer_weight": 0., "student_rollout_tokens": 32,
            "models": all_models, "errors": errors, "invalid": invalid}


def render(report):
    title = "# Linked evidence and longer on-policy SD"
    if report["mode"] == "combined":
        title += " — 1,000 cases per model"
    lines = [title, "",
             "| Model | n | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher | KV % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    for model, item in report["models"].items():
        total = item["overall"]
        a = total["accuracy"]
        lines.append(f"| {model} | {total['n']} | {fmt(a['whole'])} | {fmt(a['zero'])} | "
                     f"{fmt(a['teacher'])} | {fmt(a['cartridge'])} | "
                     f"{fmt(total['versus_teacher']['delta_pp'])} | {fmt(total['median_cartridge_percent'])} |")
    lines += ["", "| Model | Strict pairs | Linked 8-token | Linked 32-token | New right / wrong | Δ pp |",
              "|---|---:|---:|---:|---:|---:|"]
    for model, item in report["models"].items():
        total = item["overall"]
        paired = total["versus_linked8"]
        lines.append(f"| {model} | {paired['n']} | {fmt(total['paired_linked8_accuracy'])} | "
                     f"{fmt(total['paired_cartridge_accuracy'])} | {paired['new_correct']} / "
                     f"{paired['new_wrong']} | {fmt(paired['delta_pp'])} |")
    lines += ["", "| Model / question type | n | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model, item in report["models"].items():
        for index in range(1, 11):
            group = item["groups"].get("task:qa" + str(index))
            if not group:
                continue
            a = group["accuracy"]
            lines.append(f"| {model} QA{index} | {group['n']} | {fmt(a['whole'])} | {fmt(a['zero'])} | "
                         f"{fmt(a['teacher'])} | {fmt(a['cartridge'])} | "
                         f"{fmt(group['versus_teacher']['delta_pp'])} |")
    lines += ["", "The paired comparison uses the same subset for both rollout lengths. The teacher "
              "has the original generic response-format "
              "instruction. Full-context training uses selected-teacher on-policy SD only; "
              "the final cartridge is physically independent.",
              "KV % is the median materialized cartridge size relative to the full document KV; "
              "it excludes model weights, question/generated suffixes, and training peak memory.",
              "", "The 200-case configuration study and 800 source-disjoint cases are reported "
              "separately. Aggregate scores from the wider corpus have been viewed in prior "
              "experiments; this is not a completely blind holdout.", ""]
    if report["mode"] == "combined":
        for model, item in report["models"].items():
            lines.append(f"{model}: source-disjoint holdout target — {item['decision']['reason']}.")
        lines.append("")
    return "\n".join(lines)


def combine_reports(development, holdout):
    """Build an honest 200 + 800 table after all selected holdout cases finish."""
    shared = ("metric", "selection_policy", "student_rollout_tokens", "selected_transfer_weight",
              "cached_scores", "online_api_calls", "evaluation_answers_used_only_posthoc")
    if (development["mode"] != "development" or holdout["mode"] != "holdout" or
        development["expected_per_model"] != 200 or holdout["expected_per_model"] != 800 or
        any(development[key] != holdout[key] for key in shared) or
        development["errors"] or development["invalid"] or holdout["errors"] or holdout["invalid"]):
        raise ValueError("The development and holdout reports are not compatible")
    models = {}
    for model, held_item in holdout["models"].items():
        if model not in development["models"]:
            raise ValueError("A holdout model has no development run")
        dev_rows, held_rows = development["models"][model]["rows"], held_item["rows"]
        if len(dev_rows) != 200 or len(held_rows) != 800:
            return None
        validate_partition([row["case"] for row in dev_rows], [row["case"] for row in held_rows])
        rows = dev_rows + held_rows
        groups = defaultdict(list)
        for row in rows:
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        models[model] = {"overall": summarize(rows), "groups": {key: summarize(value) for key, value in groups.items()},
                         "rows": rows, "decision": held_item["decision"],
                         "development": development["models"][model]["overall"],
                         "holdout": held_item["overall"],
                         "holdout_excluding_viewed_examples": held_item["excluding_viewed_examples"]}
    if not models:
        return None
    return {**{key: development[key] for key in shared}, "mode": "combined", "expected_per_model": 1000,
            "created_utc": datetime.now(timezone.utc).isoformat(), "models": models,
            "errors": [], "invalid": []}


def write_report(report, stem):
    for suffix, text in ((".json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                         (".md", render(report))):
        output = Path(stem).with_suffix(suffix)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, action="append", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected shared evaluation metric")
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root, args.baseline_root,
                     json.loads(args.references.read_text()), score_prediction)
    write_report(report, args.output)


if __name__ == "__main__":
    main()
