#!/usr/bin/env python3
"""Decompose saved paired outcomes by support coverage, model and format."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

from experiments.babilong_linked_selection.coverage_outcomes import coverage_group

ARMS = ("whole", "baseline10", "zero", "teacher", "cartridge", "zero_teacher_format", "cartridge_teacher_format")
PAIRS = (("selection_and_method", "baseline10", "cartridge"),
         ("sd_original_format", "zero", "cartridge"),
         ("sd_teacher_format", "zero_teacher_format", "cartridge_teacher_format"),
         ("teacher_to_cartridge", "teacher", "cartridge"))


def summarize(rows):
    n = len(rows)
    accuracy = {key: 100 * sum(row[key] for row in rows) / n if n else None for key in ARMS}
    pairs = {}
    for label, before, after in PAIRS:
        gains = sum(not row[before] and row[after] for row in rows)
        losses = sum(row[before] and not row[after] for row in rows)
        pairs[label] = dict(new_correct=gains, new_wrong=losses,
                            delta_pp=100 * (gains - losses) / n if n else None)
    groups = {"whole_only": [], "teacher_only": [], "both_correct": [], "both_wrong": []}
    for row in rows:
        group = ("both_correct" if row["whole"] and row["teacher"] else
                 "whole_only" if row["whole"] else "teacher_only" if row["teacher"] else "both_wrong")
        groups[group].append(row)
    complementarity = {name: dict(n=len(group),
        cartridge_correct=sum(row["cartridge"] for row in group),
        cartridge_teacher_format_correct=sum(row["cartridge_teacher_format"] for row in group),
        new_correct_vs_zero=sum(not row["zero"] and row["cartridge"] for row in group),
        new_wrong_vs_zero=sum(row["zero"] and not row["cartridge"] for row in group))
        for name, group in groups.items()}
    return dict(n=n, accuracy=accuracy, pairs=pairs, complementarity=complementarity)


def selected_chunks(result):
    selection = result["selection"]
    if (selection["complete_original_document"] is not True or
        selection["reference_answers_used"] is not False or selection["online_api_calls"] != 0):
        raise ValueError("Expected full-source, cached-score selection without labels")
    return sorted({item for group in selection["selected_original_chunks"] for item in group})


def validate_pair(key, row, coverage, old, new, reference, grade):
    if (old["key"] != key or new["key"] != key or row["case"] != key or
        old["model"] != new["model"] or old["task"] != new["task"] or
        row["task"] != new["task"] or reference["task"] != new["task"] or
        selected_chunks(old) != coverage["score10"]["selected"] or
        selected_chunks(new) != coverage["entity10"]["selected"] or
        row["baseline_source"] != "full_1000"):
        raise ValueError("The saved model, task, case or selected chunks do not match")
    for result in (old, new):
        selection, training = result["selection"], result["training"]
        if (selection["source_case_key"] != key or training["selected_transfer_weight"] != 0 or
            len(selected_chunks(result)) != max(1, math.ceil(.1 * selection["total_chunks"])) or
            training["objective"] != "on-policy KL(teacher || student)" or
            training["student_rollout_limit"] != 8 or training["optimizer"] != "Adam" or
            result["self_study"]["reference_answers_used"] or result["self_study"]["student_answer_hint_used"] or
            result["teacher_answer_in_final_prompt"] or not result["backbone_unchanged"] or
            not result["original_document_KV_unchanged"]):
            raise ValueError("The original SD protocol changed")
        pair = result["standalone_cartridge"]["paired_prompt_control"]
        if (not pair["same_selected_kv"] or not pair["same_backbone_and_device"] or
            not pair["same_generation_limit"] or pair["control_update_steps"] != 0 or
            pair["teacher_answer_in_student_prompt"] or pair["reference_answers_used"]):
            raise ValueError("The matched-format SD controls changed")
    if old["selection"]["total_chunks"] != new["selection"]["total_chunks"]:
        raise ValueError("The selection budget differs")
    pair = new["standalone_cartridge"]["paired_prompt_control"]
    baseline = old["standalone_cartridge"]["paired_prompt_control"]
    def score(text):
        value = grade(text, reference["answers"], "babilong", reference["task"])[0]
        if type(value) not in (int, float) or value not in (0, 1):
            raise ValueError("The common judge must return a binary BABILONG score")
        return int(value)
    recomputed = dict(whole=score(new["self_judge"]["native_prediction"]),
        baseline10=score(baseline["trained_original"]["prediction"]),
        zero=score(pair["zero_original"]["prediction"]), teacher=score(new["teacher"]["prediction"]),
        cartridge=score(pair["trained_original"]["prediction"]),
        zero_teacher_format=score(pair["zero_teacher_suffix"]["prediction"]),
        cartridge_teacher_format=score(pair["trained_teacher_suffix"]["prediction"]))
    if any(row.get(name) not in (0, 1) or row[name] != value for name, value in recomputed.items()):
        raise ValueError("The cached scores differ from the common judge")
    if row["same_full_answer"] != (old["self_judge"]["native_prediction"] == new["self_judge"]["native_prediction"]):
        raise ValueError("The full-view output pairing changed")
    return dict(case=key, task=row["task"], length=row["length"],
        coverage=coverage_group(coverage["score10"]["status"], coverage["entity10"]["status"]),
        same_full_answer=row["same_full_answer"], **recomputed)


def audit(manifest, coverage, report, references, linked_root, baselines, grade):
    keys = manifest["case_ids"]
    if (len(keys) != 200 or len(set(keys)) != 200 or report["metric"] != "answer_focused_lexical_v2" or
        report["development_only"] is not True or report["expected_per_model"] != 200 or
        report["selected_fraction"] != .1 or report["baseline_fraction"] != .1 or
        report["selection_policy"] != "decision_entity_link_v1" or
        report["cached_scores"] is not True or report["online_api_calls"] != 0 or
        report["evaluation_answers_used_only_posthoc"] is not True or report["errors"] or report["invalid"]):
        raise ValueError("The fixed 200-case paired development report changed")
    annotations = {item["case"]: item["outcomes"] for item in coverage["rows"]}
    if len(coverage["rows"]) != 200 or set(annotations) != set(keys):
        raise ValueError("The annotated selection cohort changed")
    models = {}
    for model in ("qwen", "gpt"):
        saved = {row["case"]: row for row in report["models"][model]["rows"]}
        if len(report["models"][model]["rows"]) != 200 or set(saved) != set(keys):
            raise ValueError("The paired model outputs are incomplete")
        rows = []
        for key in keys:
            old = json.loads((Path(baselines[model]) / "cases" / key / "weight_0.00.json").read_text())
            new = json.loads((Path(linked_root) / "cases" / model / key / "result.json").read_text())
            rows.append(validate_pair(key, saved[key], annotations[key], old, new, references[key], grade))
        groups = defaultdict(list)
        for row in rows:
            groups["coverage:" + row["coverage"]].append(row)
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        models[model] = dict(overall=summarize(rows),
            groups={name: summarize(group) for name, group in sorted(groups.items())},
            matched_full_answers=sum(row["same_full_answer"] for row in rows))
    return dict(scope="fixed 200-case development split; 50 source questions at four lengths",
        metric=report["metric"], keep_fraction=.1, online_api_calls=0, new_model_calls=0,
        reference_answers_used_only_after_generation=True,
        teacher_has_extra_generic_format_instruction=True,
        baseline_and_linked_runs_may_use_different_gpu_allocations=True, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "coverage", "report", "references", "linked-root", "qwen-baseline",
                 "gpt-baseline", "scorer-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_root))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected scorer version")
    result = audit(json.loads(args.manifest.read_text()), json.loads(args.coverage.read_text()),
        json.loads(args.report.read_text()), json.loads(args.references.read_text()), args.linked_root,
        dict(qwen=args.qwen_baseline, gpt=args.gpt_baseline), score_prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for model, data in result["models"].items():
        print(model, json.dumps(data["overall"]))


if __name__ == "__main__":
    main()
