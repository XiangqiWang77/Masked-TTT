#!/usr/bin/env python3
"""Freeze a GPT format choice from development and report its held-out result.

This scores already generated answers; it makes no model or Decision API calls.
The generic-format answer in existing runs is a selected-KV diagnostic created
before physical release. A separate real-model check is required to establish
that the same tokens are produced after release.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition

ARMS = ("whole", "baseline10", "zero", "teacher", "cartridge", "zero_teacher_format", "cartridge_teacher_format")


def summarize(rows):
    n = len(rows)
    if any(type(row.get(key)) is not int or row[key] not in (0, 1) for row in rows for key in ARMS):
        raise ValueError("Expected binary scores for each same-run control")
    accuracy = {key: 100 * sum(row[key] for row in rows) / n if n else None for key in ARMS}
    def pair(reference):
        new = sum(row["cartridge_teacher_format"] and not row[reference] for row in rows)
        lost = sum(row[reference] and not row["cartridge_teacher_format"] for row in rows)
        return dict(new_correct=new, new_wrong=lost, delta_pp=100 * (new-lost) / n if n else None)
    return dict(n=n, accuracy=accuracy, versus_teacher=pair("teacher"),
                versus_zero_same_format=pair("zero_teacher_format"),
                versus_original_format=pair("cartridge"), versus_whole=pair("whole"))


def validate_common(manifest, report, *, mode):
    if (manifest["mode"] != mode or manifest["keep_fraction"] != .1 or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["entity_reserve_fraction"] != .25 or manifest["basis_source"] != "document" or
        manifest["selected_transfer_weight"] != 0 or manifest["online_api_calls"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or
        report["metric"] != "answer_focused_lexical_v2" or report["online_api_calls"] != 0 or
        report["cached_scores"] is not True or report["evaluation_answers_used_only_posthoc"] is not True or
        report["selection_policy"] != "decision_entity_link_v1"):
        raise ValueError("The cached-score OLMo-style protocol changed")
    if any(not isinstance(item, dict) or item.get("model") in (None, "gpt")
           for item in report["errors"] + report["invalid"]):
        raise ValueError("The GPT result has failed or invalid cases")
    cases = manifest["case_ids"]
    rows = report["models"]["gpt"]["rows"]
    if len(cases) != (200 if mode == "development" else 800) or len(set(cases)) != len(cases):
        raise ValueError("Unexpected case cohort")
    if len({row["case"] for row in rows}) != len(rows) or not {row["case"] for row in rows}.issubset(cases):
        raise ValueError("Duplicate or out-of-cohort cases")
    if any(row["baseline_source"] != "full_1000" for row in rows):
        raise ValueError("The historical paired baseline changed")
    summarize(rows)
    return rows


def freeze_development(manifest, report):
    validate_partition(manifest["case_ids"], manifest["holdout_case_ids"])
    rows = validate_common(manifest, report, mode="development")
    if len(rows) != 200 or report["development_only"] is not True:
        raise ValueError("The full 200-case development report is required")
    stats = summarize(rows)
    a = stats["accuracy"]
    if not all(a["cartridge_teacher_format"] > a[key]
               for key in ("whole", "teacher", "zero_teacher_format", "cartridge")):
        raise ValueError("The fixed generic-format candidate did not pass its development comparisons")
    return dict(model="gpt", selected_format="teacher_generic_requirements", development_cases=200,
        development_source_questions=50, holdout_cases=800, holdout_source_questions=200,
        selection_policy="decision_entity_link_v1", keep_fraction=.1,
        selection_uses_only_development_report=True, online_api_calls=0,
        metric="answer_focused_lexical_v2", development=stats,
        previously_viewed_holdout_sources=len(manifest["qualitative_case_sources"]),
        selected_format_has_teacher_answer=False,
        holdout_generic_answer_is_pre_release_diagnostic=True,
        requires_loaded_model_release_verification=True)


def evaluate(plan, dev_manifest, dev_report, held_manifest, held_report):
    if freeze_development(dev_manifest, dev_report) != plan["decision"]:
        raise ValueError("The frozen development selection changed")
    validate_partition(dev_manifest["case_ids"], held_manifest["case_ids"])
    if (held_manifest["case_ids"] != dev_manifest["holdout_case_ids"] or
        held_report["development_only"] is not False or held_report["expected_per_model"] != 800):
        raise ValueError("The source-disjoint holdout changed")
    rows = validate_common(held_manifest, held_report, mode="holdout")
    dev_rows = dev_report["models"]["gpt"]["rows"]
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    combined = summarize(dev_rows + rows) if len(rows) == 800 else None
    return dict(metric=plan["decision"]["metric"], partial=len(rows) != 800,
        selected_format=plan["decision"]["selected_format"],
        development=plan["decision"]["development"], holdout=summarize(rows),
        combined=combined, groups={key: summarize(group) for key, group in sorted(groups.items())},
        source_report_utc=datetime.fromtimestamp(held_report["updated"], timezone.utc).isoformat(),
        labels_used_only_posthoc=True, model_or_api_calls=0,
        generic_answer_is_pre_release_diagnostic=True,
        requires_loaded_model_release_verification=True,
        previously_viewed_holdout_sources=plan["decision"]["previously_viewed_holdout_sources"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path)
    parser.add_argument("--heldout-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    dev_manifest, dev_report = read(args.development_manifest), read(args.development_report)
    if args.heldout_manifest is None and args.heldout_report is None and args.output is None:
        if args.plan.exists():
            raise ValueError("Refuse to overwrite a frozen development selection")
        plan = dict(locked_utc=datetime.now(timezone.utc).isoformat(),
                    decision=freeze_development(dev_manifest, dev_report))
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan["decision"]["development"]))
        return
    if args.heldout_manifest is None or args.heldout_report is None or args.output is None:
        raise ValueError("Provide all held-out report arguments")
    result = evaluate(read(args.plan), dev_manifest, dev_report,
                      read(args.heldout_manifest), read(args.heldout_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["holdout"]))


if __name__ == "__main__":
    main()
