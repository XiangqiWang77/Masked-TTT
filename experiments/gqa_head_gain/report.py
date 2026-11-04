#!/usr/bin/env python3
"""Score the per-query-head gain experiment against its own and paired controls."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys


def validate_candidate(result, case, model, task, gain_head_mode="query",
                       selection_policy="decision_entity_link_v1"):
    selection, train = result["selection"], result["training"]
    final = result["standalone_cartridge"]
    controls, memory = final["original_question_controls"], final["memory"]
    release = result["memory"]["full_cache_release_before_standalone"]
    if (result["key"] != case or result["model_short"] != model or result["task"] != task or
        result["keep_fraction"] != .1 or selection["source_case_key"] != case or
        selection["backend"] != "precomputed_source_JSON" or
        selection["selection_policy"] != selection_policy or
        selection["entity_reserve_fraction"] != .25 or
        len(selection["selected_chunks"]) != max(1, math.ceil(selection["total_chunks"] * .1)) or
        not selection["complete_original_document"] or selection["reference_answers_used"] or
        selection["online_api_calls"] != 0 or train["gain_head_mode"] != gain_head_mode or
        train["objective"] != "on-policy KL(teacher || student)" or
        train["value_basis_source"] != "document" or train["selected_transfer_weight"] != 0 or
        train["student_rollout_limit"] != 8 or not train["steps"] or
        not train["teacher_conditioned_on_student_prefix"] or
        not train["student_rollouts_refreshed_each_step"] or
        any("selected_transfer_KL_before" in step for step in train["steps"]) or
        not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
        not isinstance(result["self_judge"]["native_prediction"], str) or
        not result["teacher"]["response_instruction_applied"] or
        result["self_study"]["reference_answers_used"] or
        result["self_study"]["student_answer_hint_used"] or
        result["teacher_answer_in_final_prompt"] or final["teacher_answer_in_final_prompt"] or
        controls["reference_answers_used"] or controls["teacher_answer_in_student_prompt"] or
        not controls["same_selected_kv"] or not controls["same_question_suffix"] or
        not controls["same_generation_limit"] or controls["teacher_suffix_matches_student"] or
        result["output_source"] != "standalone_cartridge" or result["prediction"] != final["prediction"] or
        result["generated_token_ids"] != final["generated_token_ids"] or
        controls["adapted"]["tokens"] != final["generated_token_ids"] or
        not release["requested"] or release["full_cache_tensors_live_before_standalone"] != 0 or
        not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
        memory["additional_source_kv_bytes_before_decode"] != 0 or
        memory["additional_training_adapter_bytes_before_decode"] != 0 or
        (model == "gpt" and result["request_chat_date"] != "2026-11-03")):
        raise ValueError("The original SD, selected evidence, or independent-cartridge contract changed")


def validate_pair(old, new, case):
    """Strict pairs use the same evidence, teacher, zero answer, and training prompts."""
    a, b = old["training"], new["training"]
    x, y = old["selection"], new["selection"]
    before = old["standalone_cartridge"]["paired_prompt_control"]
    after = new["standalone_cartridge"]["original_question_controls"]
    shared = ("objective", "optimizer", "teacher_cap", "teacher_target_scope",
              "adaptive_max_answer_steps", "adaptive_answer_KL_stop", "student_rollout_limit")
    practice_fields = ("question", "answer", "evidence", "source_chunk", "verifier_answer", "accepted")
    proposals = lambda item: [{key: entry[key] for key in practice_fields}
                              for entry in item["self_study"]["checks"]]
    if (old["key"] != case or new["key"] != case or old["task"] != new["task"] or
        old["model"] != new["model"] or x["selected_original_chunks"] != y["selected_original_chunks"] or
        x["total_chunks"] != y["total_chunks"] or x["original_prompt_tokens"] != y["original_prompt_tokens"] or
        old["teacher"]["tokens"] != new["teacher"]["tokens"] or
        before["zero_original"]["tokens"] != after["zero"]["tokens"] or proposals(old) != proposals(new) or
        not a["steps"] or not b["steps"] or
        a["steps"][0]["student_rollout_tokens"] != b["steps"][0]["student_rollout_tokens"] or
        any(a.get(key) != b.get(key) for key in shared) or a["student_rollout_limit"] != 8 or
        a["selected_transfer_weight"] != 0 or b["selected_transfer_weight"] != 0 or
        a.get("value_basis_source", "document") != "document" or b["value_basis_source"] != "document" or
        not before["same_selected_kv"] or not before["same_generation_limit"] or
        not after["same_selected_kv"] or not after["same_question_suffix"] or
        not after["same_generation_limit"] or before["reference_answers_used"] or
        after["reference_answers_used"]):
        raise ValueError("The shared-KV and per-query-head runs are not strictly paired")


def summarize(rows):
    keys = ("whole", "zero", "teacher", "cartridge")
    def mean(items, key):
        return 100 * sum(row[key] for row in items) / len(items) if items else None
    def change(items, previous):
        gain = sum(row["cartridge"] and not row[previous] for row in items)
        loss = sum(row[previous] and not row["cartridge"] for row in items)
        return dict(n=len(items), new_correct=gain, new_wrong=loss,
                    delta_pp=100 * (gain - loss) / len(items) if items else None)
    paired = [row for row in rows if row.get("baseline") is not None]
    sizes = [row["cartridge_percent"] for row in rows]
    return dict(n=len(rows), accuracy={key: mean(rows, key) for key in keys},
                versus_teacher=change(rows, "teacher"), versus_zero=change(rows, "zero"),
                versus_whole=change(rows, "whole"), strictly_paired=change(paired, "baseline"),
                paired_baseline_accuracy=mean(paired, "baseline"),
                paired_candidate_accuracy=mean(paired, "cartridge"),
                median_cartridge_percent=statistics.median(sizes) if sizes else None)


def collect(manifest, candidate_root, baseline_root, refs, score):
    if (manifest["case_count"] not in (200, 800) or len(manifest["case_ids"]) != manifest["case_count"] or
        len(set(manifest["case_ids"])) != manifest["case_count"] or
        manifest["gain_head_mode"] != "query" or manifest["keep_fraction"] != .1 or
        manifest["student_rollout_tokens"] != 8 or manifest["selected_transfer_weight"] != 0 or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The frozen gain comparison protocol changed")
    models, errors, invalid = {}, [], []
    for model in manifest["models"]:
        rows = []
        for case in manifest["case_ids"]:
            folder = Path(candidate_root) / "cases" / model / case
            path = folder / "result.json"
            if not path.exists():
                if (folder / "error.json").exists():
                    error = json.loads((folder / "error.json").read_text())
                    errors.append(dict(model=model, case=case, type=error.get("type")))
                continue
            result = json.loads(path.read_text())
            ref = refs[case]
            try:
                validate_candidate(result, case, model, ref["task"])
            except (ValueError, KeyError, TypeError) as exc:
                invalid.append(dict(model=model, case=case, reason=str(exc)))
                continue
            grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
            controls = result["standalone_cartridge"]["original_question_controls"]
            row = dict(case=case, task=ref["task"], length=result["length"],
                whole=grade(result["self_judge"]["native_prediction"]),
                zero=grade(controls["zero"]["prediction"]), teacher=grade(result["teacher"]["prediction"]),
                cartridge=grade(result["prediction"]), baseline=None,
                cartridge_percent=100 * result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"] /
                                  result["memory"]["full_document_kv_bytes"])
            baseline = Path(baseline_root) / "cases" / model / case / "result.json"
            if baseline.exists():
                try:
                    old = json.loads(baseline.read_text())
                    validate_pair(old, result, case)
                    row["baseline"] = grade(old["standalone_cartridge"]["paired_prompt_control"]
                                            ["trained_original"]["prediction"])
                except (ValueError, KeyError, TypeError) as exc:
                    row["pair_rejected"] = str(exc)
            rows.append(row)
        groups = defaultdict(list)
        for row in rows:
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        overall = summarize(rows)
        enough = overall["n"] == manifest["case_count"]
        gains = {name: overall["versus_" + name]["delta_pp"] for name in ("whole", "zero", "teacher")}
        eligible = enough and not any(x["model"] == model for x in errors + invalid) and all(x > 0 for x in gains.values())
        models[model] = dict(overall=overall, groups={key: summarize(value) for key, value in groups.items()},
                             rows=rows, development_gate=dict(eligible=eligible, delta_pp=gains))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), metric="answer_focused_lexical_v2",
                expected_per_model=manifest["case_count"], cached_scores=True, online_api_calls=0,
                evaluation_answers_used_only_posthoc=True, gain_head_mode="query", models=models,
                errors=errors, invalid=invalid)


def render(report):
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    lines = ["# Query-head gain: same-method development comparison", "",
             "| Model | n | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model, item in report["models"].items():
        stats = item["overall"]; acc = stats["accuracy"]
        lines.append(f"| {model} | {stats['n']} | {fmt(acc['whole'])} | {fmt(acc['zero'])} | "
                     f"{fmt(acc['teacher'])} | {fmt(acc['cartridge'])} | {fmt(stats['versus_teacher']['delta_pp'])} |")
    lines += ["", "| Model | Strict pairs | Shared-KV gain | Query-head gain | New right / wrong | Δ pp |",
              "|---|---:|---:|---:|---:|---:|"]
    for model, item in report["models"].items():
        stats = item["overall"]; pair = stats["strictly_paired"]
        lines.append(f"| {model} | {pair['n']} | {fmt(stats['paired_baseline_accuracy'])} | "
                     f"{fmt(stats['paired_candidate_accuracy'])} | {pair['new_correct']} / {pair['new_wrong']} | "
                     f"{fmt(pair['delta_pp'])} |")
    lines += ["", "Teacher and Student use the original on-policy KL, cached 10% evidence and the "
              "same eight-token rollout limit; only the gain head-sharing changes. The Student's selected-only "
              "cartridge is physically released from all other document caches before decode.",
              "Scores use the same task-aware loose lexical judge; reference answers are used only after generation.",
              "Strict pairs require identical selected evidence, Teacher tokens, zero-update answers and practice prompts.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "candidate-root", "baseline-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected evaluation metric")
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root, args.baseline_root,
                     json.loads(args.references.read_text()), score_prediction)
    for suffix, text in ((".json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                         (".md", render(report))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
