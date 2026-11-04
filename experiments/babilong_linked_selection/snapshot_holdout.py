#!/usr/bin/env python3
"""Freeze a partial source-disjoint holdout report without case identities."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.babilong_linked_selection.paired_attribution import ARMS, summarize


def source_id(case):
    fields = case.split("-", 2)
    if len(fields) != 3 or fields[0] != "babilong" or fields[1] not in ("8", "16", "32", "64"):
        raise ValueError("Unexpected BABILONG case identity")
    return fields[2]


def freeze(holdout, development, report):
    holdout_keys, dev_keys = holdout["case_ids"], development["case_ids"]
    if (holdout["mode"] != "holdout" or holdout["case_count"] != 800 or len(holdout_keys) != 800 or
        len(set(holdout_keys)) != 800 or len(dev_keys) != 200 or len(set(dev_keys)) != 200 or
        holdout["models"] != ["qwen", "gpt"] or holdout["keep_fraction"] != .1 or
        holdout["selection_policy"] != "decision_entity_link_v1" or holdout["entity_reserve_fraction"] != .25 or
        holdout["basis_source"] != "document" or
        holdout["reference_answers_used_in_inference"] is not False or holdout["online_api_calls"] != 0 or
        holdout["selected_transfer_weight"] != 0 or holdout["holdout_evaluated_at_selection"] is not False or
        report["metric"] != "answer_focused_lexical_v2" or report["development_only"] is not False or
        report["expected_per_model"] != 800 or report["selection_policy"] != "decision_entity_link_v1" or
        report["selected_fraction"] != .1 or report["baseline_fraction"] != .1 or report["cached_scores"] is not True or
        report["online_api_calls"] != 0 or report["evaluation_answers_used_only_posthoc"] is not True):
        raise ValueError("The frozen source-disjoint holdout protocol changed")
    holdout_sources = {source_id(key) for key in holdout_keys}
    dev_sources = {source_id(key) for key in dev_keys}
    if len(holdout_sources) != 200 or len(dev_sources) != 50 or holdout_sources & dev_sources:
        raise ValueError("Development and holdout sources are not disjoint")
    prior = set(holdout["qualitative_case_sources"])
    if len(prior) != 2 or not prior.issubset(holdout_sources):
        raise ValueError("Previously viewed sources must be identified")
    models = {}
    for model in ("qwen", "gpt"):
        rows = report["models"][model]["rows"]
        if (len({row["case"] for row in rows}) != len(rows) or
            not {row["case"] for row in rows}.issubset(holdout_keys)):
            raise ValueError("The held-out output cases are duplicated or out of scope")
        for row in rows:
            if (row["baseline_source"] != "full_1000" or type(row["same_full_answer"]) is not bool or
                any(type(row.get(name)) is not int or row[name] not in (0, 1) for name in ARMS)):
                raise ValueError("A paired holdout score is missing or invalid")
        grouped = defaultdict(list)
        for row in rows:
            grouped["task:" + row["task"]].append(row)
            grouped["length:" + str(row["length"])].append(row)
        untouched = [row for row in rows if source_id(row["case"]) not in prior]
        models[model] = dict(overall=summarize(rows),
            whole_accuracy=100 * sum(row["whole"] for row in rows) / len(rows) if rows else None,
            groups={name: summarize(items) for name, items in sorted(grouped.items())},
            source_questions_observed=len({source_id(row["case"]) for row in rows}),
            matched_full_answers=sum(row["same_full_answer"] for row in rows),
            previously_viewed_cases=len(rows)-len(untouched),
            excluding_previously_viewed_sources=summarize(untouched))
    return dict(scope="partial 800-case source-disjoint holdout; no retuning",
        source_report_utc=datetime.fromtimestamp(report["updated"], timezone.utc).isoformat(),
        metric=report["metric"], expected_per_model=800, development_sources=50, holdout_sources=200,
        keep_fraction=.1, selector="decision_entity_link_v1", teacher_has_extra_generic_format_instruction=True,
        model_or_api_calls=0, labels_used_only_after_generation=True,
        errors=len(report["errors"]), invalid=len(report["invalid"]), models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("holdout", "development", "report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    data = freeze(json.loads(args.holdout.read_text()), json.loads(args.development.read_text()),
                  json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for model, value in data["models"].items():
        print(model, value["overall"]["n"], json.dumps(value["overall"]["accuracy"]))


if __name__ == "__main__":
    main()
