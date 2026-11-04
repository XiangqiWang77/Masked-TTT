#!/usr/bin/env python3
"""Rescore and freeze the completed paired precision pilot without case data."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.precision_cartridge.benchmark_report import collect, summarize


def token_changes(root, cases, model):
    counts = dict(cases=0, trained_sequences_changed=0, trained_first_token_changed=0,
                  zero_sequences_changed=0, zero_first_token_changed=0)
    for case in cases:
        result = json.loads((root / "cases" / model / case / "result.json").read_text())
        final = result["standalone_cartridge"]
        diagnostic = final["precision_paired_diagnostic"]
        original, other = final["generated_token_ids"], diagnostic["precision_tokens"]
        zero, other_zero = final["original_question_controls"]["zero"]["tokens"], diagnostic["precision_zero"]["tokens"]
        if any(not isinstance(tokens, list) or not tokens for tokens in (original, other, zero, other_zero)):
            raise ValueError("Missing paired generated token sequences")
        counts["cases"] += 1
        counts["trained_sequences_changed"] += original != other
        counts["trained_first_token_changed"] += original[0] != other[0]
        counts["zero_sequences_changed"] += zero != other_zero
        counts["zero_first_token_changed"] += zero[0] != other_zero[0]
    return counts


def freeze(manifest, finished, saved, fresh):
    cases = manifest["case_ids"]
    if (manifest["case_count"] != 56 or len(cases) != 56 or len(set(cases)) != 56 or
        manifest["models"] != ["qwen", "gpt"] or manifest["keep_fraction"] != .1 or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["entity_reserve_fraction"] != .25 or manifest["student_rollout_tokens"] != 32 or
        manifest["base_steps"] != 2 or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        manifest["precision_comparison"] != "same_trained_state_bf16_fp32" or
        not finished["all_complete"] or finished["completed"] != {"qwen": 56, "gpt": 56}):
        raise ValueError("The completed 56-case paired protocol changed")
    expected = set(cases)
    grouped = {}
    for case in cases:
        prefix, length, source = case.split("-", 2)
        if prefix != "babilong" or length not in ("8", "16", "32", "64"):
            raise ValueError("Unexpected case identity")
        grouped.setdefault(source, set()).add(length)
    if len(grouped) != 14 or any(lengths != {"8", "16", "32", "64"} for lengths in grouped.values()):
        raise ValueError("Each of the 14 source questions must have four lengths")
    for report in (saved, fresh):
        if (report["metric"] != "answer_focused_lexical_v2" or report["expected_per_model"] != 56 or
            report["cached_scores"] is not True or report["online_api_calls"] != 0 or
            report["evaluation_answers_used_only_posthoc"] is not True or
            report["errors"] or report["invalid"] or set(report["models"]) != {"qwen", "gpt"}):
            raise ValueError("The paired report is incomplete or uses a different scorer")
    models = {}
    for model in ("qwen", "gpt"):
        prior, rerun = saved["models"][model], fresh["models"][model]
        old_rows, new_rows = prior["rows"], rerun["rows"]
        if (len(old_rows) != 56 or len(new_rows) != 56 or
            {row["case"] for row in old_rows} != expected or
            {row["case"] for row in new_rows} != expected or
            sorted(old_rows, key=lambda row: row["case"]) != sorted(new_rows, key=lambda row: row["case"]) or
            prior["overall"] != summarize(old_rows) or rerun["overall"] != summarize(new_rows) or
            prior["groups"] != rerun["groups"]):
            raise ValueError("The archived answers and independent rescoring differ")
        models[model] = {"overall": rerun["overall"], "groups": rerun["groups"]}
    return dict(metric="answer_focused_lexical_v2", cases_per_model=56, source_questions=14,
        four_lengths_per_source=True, selected_fraction=.1, selection_policy="decision_entity_link_v1",
        student_rollout_tokens=32, base_steps=2, models=models,
        full_student_selected_teacher_on_policy_kl=True, reference_answers_used_only_posthoc=True,
        same_training_state_for_both_cartridges=True, independently_rescored=True,
        bf16_attention_backend="normal", fp32_attention_backend="eager",
        paired_run_retains_bf16_during_fp32_decode=True, online_api_calls=0, new_model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "query_cartridges_precision_pilot_20261104"
    manifest, finished, saved = (json.loads((root / name).read_text())
                                 for name in ("manifest.json", "finished.json", "latest.json"))
    references = json.loads((workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json").read_text())
    sys.path.insert(0, str(root / "runtime/scorer"))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected scorer version")
    fresh = collect(manifest, root, references, score_prediction)
    result = freeze(manifest, finished, saved, fresh)
    for model in result["models"]:
        result["models"][model]["token_changes"] = token_changes(root, manifest["case_ids"], model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for model, item in result["models"].items():
        print(model, json.dumps(item["overall"]))


if __name__ == "__main__":
    main()
