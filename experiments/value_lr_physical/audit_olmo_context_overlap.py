#!/usr/bin/env python3
"""Compare the same BABILONG cases by OLMo working-document coverage."""
import argparse
import csv
import json
from pathlib import Path


def paired(rows, teacher_key="teacher", answer_key="cartridge"):
    if not rows:
        raise ValueError("A comparison group cannot be empty")
    pairs = [(int(row[teacher_key]), int(row[answer_key])) for row in rows]
    if any(a not in (0, 1) or b not in (0, 1) for a, b in pairs):
        raise ValueError("Answer scores must be binary")
    n = len(pairs)
    gains = sum(a == 0 and b == 1 for a, b in pairs)
    losses = sum(a == 1 and b == 0 for a, b in pairs)
    return dict(n=n, baseline_accuracy=100 * sum(a for a, _ in pairs) / n,
                cartridge_accuracy=100 * sum(b for _, b in pairs) / n,
                new_correct=gains, new_wrong=losses,
                delta_points=100 * (gains - losses) / n)


def read_report(path):
    report = json.loads(path.read_text())
    if report["metric"] != "answer_focused_lexical_v2" or report["completed"] != 1000:
        raise ValueError("Expected a completed paired 1,000-case report")
    rows = {row["case"]: row for row in report["rows"]}
    if len(rows) != 1000:
        raise ValueError("Duplicate or missing paired cases")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmo-snapshot", type=Path, required=True)
    parser.add_argument("--qwen-report", type=Path, required=True)
    parser.add_argument("--gpt-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.olmo_snapshot.open(newline="") as stream:
        olmo = list(csv.DictReader(stream))
    if len(olmo) != 1000 or len({x["case"] for x in olmo}) != 1000:
        raise ValueError("Expected the original 1,000-case OLMo snapshot")
    others = {"qwen": read_report(args.qwen_report), "gpt": read_report(args.gpt_report)}
    if any(set(rows) != {x["case"] for x in olmo} for rows in others.values()):
        raise ValueError("Models must be compared on identical case IDs")
    data = dict(metric="answer_focused_lexical_v2", online_api_calls=0, new_model_calls=0,
                evaluation_answers_used_only_posthoc=True,
                teacher_and_original_cartridge_formats_differ=True,
                default_baseline="selected ICL Teacher with generic format instruction",
                same_format_training_baseline="selected zero-update with teacher-format instruction",
                groups={})
    for coverage, expected in (("complete", 838), ("compacted", 162)):
        group = [row for row in olmo if row["working_document"] == coverage]
        if len(group) != expected:
            raise ValueError("The OLMo working-document coverage snapshot changed")
        values = {"olmo": paired(group)}
        for model, rows in others.items():
            aligned = [rows[row["case"]] for row in group]
            values[model] = paired(aligned)
            values[model]["same_format_training"] = paired(
                aligned, "selective_teacher_format", "cartridge_teacher_format")
        data["groups"][coverage] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["groups"], indent=2))


if __name__ == "__main__":
    main()
