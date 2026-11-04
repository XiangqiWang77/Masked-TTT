#!/usr/bin/env python3
"""Score fixed-LR standalone cartridges only after generation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import (
    holdout_decision, summarize, validate_partition,
)
from experiments.value_lr_physical.protocol import development_choice, name_for_lr


MODELS = {"qwen": "Qwen/Qwen3-4B-Instruct-2507", "gpt": "openai/gpt-oss-20b"}
INSPECTED = {"qa1-300023", "qa2-300037"}


def validate_result(result, case, model, task, fixed_lr):
    training, selection = result["training"], result["selection"]
    final = result["standalone_cartridge"]
    controls, memory = final["original_question_controls"], final["memory"]
    release = result["memory"]["full_cache_release_before_standalone"]
    expected = max(1, math.ceil(selection["total_chunks"] * .1))
    if (result["key"] != case or result["model"] != MODELS[model] or result["task"] != task or
        result["keep_fraction"] != .1 or selection["source_case_key"] != case or
        selection["backend"] != "precomputed_source_JSON" or
        len(selection["selected_chunks"]) != expected or not selection["complete_original_document"] or
        selection["reference_answers_used"] or selection["online_api_calls"] != 0 or
        training["objective"] != "on-policy KL(teacher || student)" or
        training["value_basis_source"] != "document" or training["selected_transfer_weight"] != 0 or
        training["gain_head_mode"] != "kv" or training["fixed_value_learning_rate"] != fixed_lr or
        training["student_rollout_limit"] != 8 or len(training["steps"]) != 2 + result["self_study"]["verified"] or
        not training["teacher_conditioned_on_student_prefix"] or not training["student_rollouts_refreshed_each_step"] or
        any("selected_transfer_KL_before" in step for step in training["steps"]) or
        not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
        not result["teacher"]["response_instruction_applied"] or
        result["self_study"]["reference_answers_used"] or result["self_study"]["student_answer_hint_used"] or
        result["teacher_answer_in_final_prompt"] or final["teacher_answer_in_final_prompt"] or
        controls["reference_answers_used"] or controls["teacher_answer_in_student_prompt"] or
        not controls["same_selected_kv"] or not controls["same_question_suffix"] or
        not controls["same_generation_limit"] or controls["teacher_suffix_matches_student"] or
        not isinstance(result["self_judge"]["native_prediction"], str) or
        result["output_source"] != "standalone_cartridge" or result["prediction"] != final["prediction"] or
        result["generated_token_ids"] != final["generated_token_ids"] or
        controls["adapted"]["tokens"] != final["generated_token_ids"] or
        not release["requested"] or release["full_cache_tensors_live_before_standalone"] != 0 or
        not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
        memory["additional_source_kv_bytes_before_decode"] != 0 or
        memory["additional_training_adapter_bytes_before_decode"] != 0 or
        (model == "gpt" and result["request_chat_date"] != "2026-11-03")):
        raise ValueError("Original SD, fixed LR or physical standalone contract failed")


def collect(manifest, root, references, score):
    development, holdout = manifest["development_case_ids"], manifest["holdout_case_ids"]
    validate_partition(development, holdout)
    if (manifest["mode"] != "holdout" or manifest["case_count"] != 800 or
        set(manifest["case_ids"]) != set(holdout) or len(manifest["models"]) != 1 or
        manifest["models"][0] not in MODELS or manifest["selection_policy"] != "cached_decision_score10" or
        manifest["fixed_value_learning_rate"] not in (.0001, .0003, .001) or
        manifest["student_rollout_tokens"] != 8 or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The frozen physical holdout changed")
    model = manifest["models"][0]
    fixed_lr = manifest["fixed_value_learning_rate"]
    rows, errors, invalid = [], [], []
    for case in manifest["case_ids"]:
        folder = Path(root) / "cases" / model / case
        path = folder / "result.json"
        if not path.exists():
            if (folder / "error.json").exists():
                error = json.loads((folder / "error.json").read_text())
                errors.append(dict(case=case, type=error.get("type")))
            continue
        result, ref = json.loads(path.read_text()), references[case]
        try:
            validate_result(result, case, model, ref["task"], fixed_lr)
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(dict(case=case, reason=str(exc)))
            continue
        grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
        controls = result["standalone_cartridge"]["original_question_controls"]
        full_bytes = result["memory"]["full_document_kv_bytes"]
        card_bytes = result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]
        if not full_bytes or not card_bytes:
            raise ValueError("Missing materialized cache size")
        rows.append(dict(case=case, task=result["task"], length=result["length"],
            whole=grade(result["self_judge"]["native_prediction"]), zero=grade(controls["zero"]["prediction"]),
            teacher=grade(result["teacher"]["prediction"]), cartridge=grade(result["prediction"]),
            linked8=None, cartridge_percent=100 * card_bytes / full_bytes))
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    strict = [row for row in rows if row["case"].split("-", 2)[2] not in INSPECTED]
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode="holdout", model=model,
        fixed_value_learning_rate=fixed_lr, expected=800, metric="answer_focused_lexical_v2",
        cached_scores=True, online_api_calls=0, reference_answers_used_only_posthoc=True,
        overall=summarize(rows), groups={key: summarize(value) for key, value in groups.items()},
        strict_holdout=summarize(strict), decision=holdout_decision(summarize(strict)),
        rows=rows, errors=errors, invalid=invalid)


def combine(development_manifest, development, manifest, holdout):
    model, fixed_lr = manifest["models"][0], manifest["fixed_value_learning_rate"]
    choice = development_choice(development_manifest, development, model)
    if (choice["value_learning_rate"] != fixed_lr or holdout["model"] != model or
        holdout["fixed_value_learning_rate"] != fixed_lr or holdout["expected"] != 800 or
        holdout["errors"] or holdout["invalid"]):
        raise ValueError("The holdout no longer matches the development-fixed choice")
    dev_rows = []
    for row in development["models"][model]["rows"]:
        dev_rows.append(dict(case=row["case"], task=row["task"], length=row["length"],
            whole=row["whole"], zero=row["zero"], teacher=row["teacher"],
            cartridge=row[name_for_lr(fixed_lr)], linked8=None, cartridge_percent=None))
    held_rows = holdout["rows"]
    if len(dev_rows) != 200 or len(held_rows) != 800:
        return None
    validate_partition([row["case"] for row in dev_rows], [row["case"] for row in held_rows])
    rows = dev_rows + held_rows
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode="combined", model=model,
        fixed_value_learning_rate=fixed_lr, expected=1000, metric=holdout["metric"],
        overall=summarize(rows), development=summarize(dev_rows), holdout=holdout["overall"],
        strict_holdout=holdout["strict_holdout"], decision=holdout["decision"],
        groups={key: summarize(value) for key, value in groups.items()}, rows=rows)


def render(report):
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    groups = [(report["model"], report["overall"])]
    groups += [(report["model"] + " " + key[5:], value) for key, value in sorted(report["groups"].items())
               if key.startswith("task:")]
    groups += [(report["model"] + " " + str(int(key[7:]) // 1024) + "K", value)
               for key, value in sorted(report["groups"].items()) if key.startswith("length:")]
    lines = ["# Original-SD physical Query Cartridge", "",
        "| Model / question type | n | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in groups:
        a = item["accuracy"]
        lines.append(f"| {name} | {item['n']} | {fmt(a['whole'])} | {fmt(a['zero'])} | "
            f"{fmt(a['teacher'])} | {fmt(a['cartridge'])} | {fmt(item['versus_teacher']['delta_pp'])} |")
    overall = report["overall"]
    size = report["holdout"]["median_cartridge_percent"] if report["mode"] == "combined" else overall["median_cartridge_percent"]
    lines += ["", f"Fixed Value learning rate: {report['fixed_value_learning_rate']:g}. "
        f"Against selected zero: {overall['versus_zero']['new_correct']} new correct / "
        f"{overall['versus_zero']['new_wrong']} new wrong. "
        f"Against Teacher: {overall['versus_teacher']['new_correct']} new correct / "
        f"{overall['versus_teacher']['new_wrong']} new wrong.",
        (f"Holdout median standalone document KV: {size:.2f}% of whole document KV "
         "(excluding model weights, suffix KV and training peak)." if size is not None else
         "Holdout KV size is pending."),
        f"The 792-case source-disjoint target: {report['decision']['reason']}.",
        "The same loose task-aware lexical score is used for all arms. The Teacher retains its "
        "original generic response-format instruction; zero-update and Cartridge use the same "
        "original question. Full Student SD uses the on-policy selected Teacher and no extra "
        "transfer loss. The final holdout answer is decoded after full KV, source selected KV "
        "and training state are released.",
        "The learning rate was fixed from 200 development cases before the 800 source-disjoint "
        "cases. Earlier aggregate results have been seen, so the holdout is not fully blind.", ""]
    return "\n".join(lines)


def write(report, stem):
    for suffix, text in ((".json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                         (".md", render(report))):
        path = Path(stem).with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidate-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected shared scorer")
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root,
                     json.loads(args.references.read_text()), score_prediction)
    write(report, args.output)


if __name__ == "__main__":
    main()
