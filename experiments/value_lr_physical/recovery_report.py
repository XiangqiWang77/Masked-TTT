#!/usr/bin/env python3
"""Post-generation whole/selected recovery from an existing paired report.

The scorer has already run; this reader never calls a model, scheduler or API
and never exposes individual case IDs or answer text in its output.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(rows):
    fields = ("whole", "zero", "teacher", "cartridge")
    if any(type(row[key]) is not int or row[key] not in (0, 1) for row in rows for key in fields):
        raise ValueError("Paired scores must be binary integers")
    count = len(rows)
    full_only = [row for row in rows if row["whole"] and not row["zero"]]
    selected_only = [row for row in rows if row["zero"] and not row["whole"]]
    fixed = sum(row["cartridge"] for row in full_only)
    retained = sum(row["cartridge"] for row in selected_only)
    change = lambda base: dict(new_correct=sum(not row[base] and row["cartridge"] for row in rows),
        new_wrong=sum(row[base] and not row["cartridge"] for row in rows),
        delta_points=(100 * sum(row["cartridge"] - row[base] for row in rows) / count
                      if count else None))
    return dict(n=count, accuracy={key: 100 * sum(row[key] for row in rows) / count if count else None
                                  for key in fields},
        whole_only=dict(total=len(full_only), recovered=fixed,
                        percent=100 * fixed / len(full_only) if full_only else None),
        selected_only=dict(total=len(selected_only), retained=retained,
                           percent=100 * retained / len(selected_only) if selected_only else None),
        versus_zero=change("zero"), versus_teacher=change("teacher"))


def collect(report):
    mode, model, rows = report["mode"], report["model"], report["rows"]
    expected = {"holdout": 800, "combined": 1000}.get(mode)
    if (expected is None or report["expected"] != expected or model not in ("qwen", "gpt") or
        report["metric"] != "answer_focused_lexical_v2" or report["overall"]["n"] != len(rows) or
        len({row["case"] for row in rows}) != len(rows) or len(rows) > expected):
        raise ValueError("The input is not the expected paired physical-cartridge report")
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    return dict(mode=mode, model=model, expected=expected, fixed_value_learning_rate=report["fixed_value_learning_rate"],
                metric=report["metric"], full_and_selected_same_original_question=True,
                teacher_has_additional_generic_format_instruction=True,
                reference_answers_used_only_after_generation=True,
                online_api_calls=0, overall=summarize(rows),
                by_task={key: summarize(value) for key, value in sorted(groups.items())},
                error_count=len(report.get("errors", [])), invalid_count=len(report.get("invalid", [])))


def render(data):
    overall = data["overall"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}%"
    lines = [f"# {data['model']}: Whole-to-Cartridge paired outcomes", "",
        f"{overall['n']} / {data['expected']} paired cases; scope: {data['mode']}.", "",
        "| Group | n | Whole | Selected zero | ICL Teacher | Cartridge | Whole-only recovered | Selected-only retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, item in [("All", overall), *sorted(data["by_task"].items())]:
        accuracy = item["accuracy"]
        left, right = item["whole_only"], item["selected_only"]
        lines.append(f"| {label} | {item['n']} | {fmt(accuracy['whole'])} | {fmt(accuracy['zero'])} | "
            f"{fmt(accuracy['teacher'])} | {fmt(accuracy['cartridge'])} | "
            f"{left['recovered']} / {left['total']} | {right['retained']} / {right['total']} |")
    lines += ["", f"Cartridge versus selected zero: {overall['versus_zero']['new_correct']} new correct / "
              f"{overall['versus_zero']['new_wrong']} new wrong. "
              f"Versus Teacher: {overall['versus_teacher']['new_correct']} / {overall['versus_teacher']['new_wrong']}.",
              "Whole-only means Whole was correct and selected zero was wrong; selected-only is the reverse. "
              "This is post-generation answer analysis, not direct proof that omitted facts were encoded.",
              "The Teacher has a generic output-format instruction; Whole, selected zero and Cartridge use "
              "the original question format. All arms use the same task-aware loose lexical score.",
              f"Input validation errors: {data['error_count']}; invalid cases: {data['invalid_count']}.", ""]
    return "\n".join(lines)


def write(data, stem):
    for suffix, text in ((".json", json.dumps(data, indent=2) + "\n"), (".md", render(data))):
        path = Path(stem).with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.job_root.resolve()
    source = root / "combined_1000.json"
    if not source.exists():
        source = root / "latest.json"
    if not source.exists():
        print("Waiting for the paired report")
        return
    data = collect(json.loads(source.read_text()))
    write(data, root / "recovery_latest")
    print(render(data))


if __name__ == "__main__":
    main()
