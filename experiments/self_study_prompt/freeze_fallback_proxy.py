#!/usr/bin/env python3
"""Audit a conditional study fallback from already completed paired results."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(rows):
    columns = ("baseline", "cartridge", "teacher", "baseline_verified", "candidate_verified")
    if any(type(row.get(key)) is not int or row[key] < 0 for row in rows for key in columns):
        raise ValueError("Missing paired answer or practice counts")
    if any(row[key] not in (0, 1) for row in rows for key in ("baseline", "cartridge", "teacher")):
        raise ValueError("Expected binary post-generation scores")
    changed = [row for row in rows if row["baseline_verified"] == 0 and row["candidate_verified"] > 0]
    original_practices = sum(row["baseline_verified"] for row in rows)
    extra_practices = sum(row["candidate_verified"] for row in changed)
    proxy = [row["cartridge"] if row in changed else row["baseline"] for row in rows]
    gains = sum(row["cartridge"] and not row["baseline"] for row in changed)
    losses = sum(row["baseline"] and not row["cartridge"] for row in changed)
    return dict(cases=len(rows), source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
        original_verified_practices=original_practices,
        potential_additional_verified_practices=extra_practices,
        original_cases_with_practice=sum(row["baseline_verified"] > 0 for row in rows),
        cases_with_candidate_fallback=len(changed),
        fallback_source_questions=len({row["case"].split("-", 2)[2] for row in changed}),
        fallback_cases_by_task=dict(sorted(Counter(row["task"] for row in changed).items())),
        fallback_cases_candidate_vs_original=dict(new_correct=gains, new_wrong=losses),
        original_correct=sum(row["baseline"] for row in rows), proxy_correct=sum(proxy),
        teacher_correct=sum(row["teacher"] for row in rows))


def collect(path):
    report = json.loads(path.read_text())
    if (report.get("mode") != "development" or report.get("model") != "qwen" or
        report.get("expected") != 200 or report.get("metric") != "answer_focused_lexical_v2" or
        report.get("reference_answers_used_only_posthoc") is not True or report.get("online_api_calls") != 0 or
        report.get("errors") or report.get("invalid") or len(report["rows"]) != 200 or
        len({row["case"] for row in report["rows"]}) != 200 or
        report["overall"]["strictly_paired"]["n"] != 200):
        raise ValueError("Expected the complete same-GPU 200-case study-prompt comparison")
    return dict(summary=summarize(report["rows"]), proxy_from_saved_answers_not_new_inference=True,
        model_calls=0, online_api_calls=0, reference_answers_used_only_for_post_generation_scoring=True,
        fallback_not_implemented_or_submitted=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
