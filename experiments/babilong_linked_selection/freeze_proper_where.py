#!/usr/bin/env python3
"""Freeze an aggregate-only summary of a completed proper-where comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(report):
    if (report.get("mode") != "development" or report.get("model") != "qwen" or
        report.get("metric") != "answer_focused_lexical_v2" or
        report.get("expected") != 200 or report.get("online_api_calls") != 0 or
        report.get("reference_answers_used_only_posthoc") is not True or
        report.get("errors") or report.get("invalid") or report.get("unmatched")):
        raise ValueError("Unexpected development protocol or incomplete report")
    rows = report["rows"]
    if len(rows) != 200 or len({r["case"] for r in rows}) != 200:
        raise ValueError("Expected 200 distinct completed cases")
    methods = ("whole", "zero", "teacher", "cartridge", "old_whole", "old_zero", "old_teacher", "old_cartridge")
    if any(r.get(k) not in (0, 1) for r in rows for k in methods):
        raise ValueError("Missing or invalid grades")
    if any(type(r.get("selection_changed")) is not bool for r in rows):
        raise ValueError("Missing selection-change flags")
    def source(row):
        parts = row["case"].split("-", 2)
        if len(parts) != 3:
            raise ValueError("Unexpected case identifier")
        return parts[2]
    def group(subset):
        n = len(subset)
        correct = {m: sum(r[m] for r in subset) for m in methods}
        result = {"cases": n, "source_questions": len({source(r) for r in subset}),
                  "correct": correct,
                  "accuracy_percent": {m: round(correct[m] * 100 / n, 4) if n else None for m in methods}}
        for candidate, baseline in (("cartridge", "teacher"), ("cartridge", "zero"),
                                    ("cartridge", "old_cartridge"), ("teacher", "old_teacher")):
            gained = [r for r in subset if r[candidate] and not r[baseline]]
            lost = [r for r in subset if r[baseline] and not r[candidate]]
            result[f"{candidate}_vs_{baseline}"] = {
                "new_correct": len(gained), "new_wrong": len(lost),
                "new_correct_sources": len({source(r) for r in gained}),
                "new_wrong_sources": len({source(r) for r in lost}),
                "delta_pp": round((len(gained)-len(lost))*100/n, 4) if n else None}
        return result
    changed = [r for r in rows if r["selection_changed"]]
    unchanged = [r for r in rows if not r["selection_changed"]]
    if len(changed) != report.get("expected_development_changed_selections"):
        raise ValueError("Changed-selection count differs from the frozen protocol")
    return {"metric": report["metric"], "model": report["model"],
            "overall": group(rows), "changed": group(changed), "unchanged": group(unchanged),
            "changed_by_task": dict(sorted(Counter(r["task"] for r in changed).items())),
            "changed_by_length": dict(sorted(Counter(str(r["length"]) for r in changed).items())),
            "same_outputs": {k: sum(r[k] for r in rows) for k in
                             ("same_whole_output", "same_teacher_tokens", "same_zero_tokens", "same_cartridge_tokens")},
            "exploratory_selection_designed_on_development_data": True,
            "cross_run_hardware_may_differ": True,
            "teacher_and_cartridge_question_formats_differ": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = summarize(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"overall": data["overall"], "changed": data["changed"]}, indent=2))


if __name__ == "__main__":
    main()
