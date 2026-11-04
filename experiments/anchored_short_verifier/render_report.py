"""Render a post-generation, paired BABILONG report from frozen scored rows.

The input is the saved aggregate report, not raw prompts, model outputs or
reference answers. It therefore cannot influence selection or adaptation.
"""
import argparse
import json
from pathlib import Path


ARMS = ("whole", "zero", "teacher", "cartridge")
LABELS = ("Full KV", "Selected KV", "ICL Teacher", "Query Cartridge")
SOURCE = "answer_focused_lexical_v2"


def _rate(rows, name):
    values = [row[name] for row in rows]
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError(f"{name} requires binary, already-scored answers")
    return 100.0 * sum(values) / len(values) if values else None


def _pair(rows, reference):
    matched = [row for row in rows if row.get(reference) is not None]
    before = [_validate_reference(row[reference]) for row in matched]
    after = [_validate_reference(row["cartridge"]) for row in matched]
    gains = sum(new == 1 and old == 0 for old, new in zip(before, after))
    losses = sum(new == 0 and old == 1 for old, new in zip(before, after))
    delta = 100.0 * (gains - losses) / len(matched) if matched else None
    return len(matched), gains, losses, delta


def _validate_reference(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Pairwise comparison requires binary scored answers")
    return value


def _pct(value):
    return "—" if value is None else f"{value:.2f}"


def _task_key(value):
    text = str(value)
    return (int(text[2:]), text) if text.startswith("qa") and text[2:].isdigit() else (10**6, text)


def _length_key(value):
    text = str(value)
    return (int(text), text) if text.isdigit() else (10**6, text)


def _grouped(rows, field):
    result = {}
    for row in rows:
        result.setdefault(str(row[field]), []).append(row)
    return result


def _table(rows, group, sort_key):
    sections = _grouped(rows, group)
    columns = "| " + ("Task" if group == "task" else "Context") + " | n | " + " | ".join(LABELS)
    columns += " | vs Teacher: new right / new wrong | Δ pp | vs Selected: new right / new wrong | Δ pp |"
    table = [
        columns,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(sections, key=sort_key):
        subset = sections[name]
        _, gt, lt, dt = _pair(subset, "teacher")
        _, gz, lz, dz = _pair(subset, "zero")
        rates = " | ".join(_pct(_rate(subset, arm)) for arm in ARMS)
        if group == "task":
            display = name
        elif name.isdigit() and int(name) >= 1024 and int(name) % 1024 == 0:
            display = f"{int(name) // 1024}K"
        else:
            display = name
        table.append(
            f"| {display} | {len(subset)} | {rates} | {gt} / {lt} | {_pct(dt)} "
            f"| {gz} / {lz} | {_pct(dz)} |"
        )
    return table


def render(result):
    if result.get("metric") != SOURCE or not result.get("evaluation_answers_used_only_posthoc"):
        raise ValueError("Expected post-generation BABILONG scores using the fixed loose scorer")
    lines = [
        "# Query Cartridge: paired BABILONG results",
        "",
        f"Scope: {result['mode']}; expected {result['expected_per_model']} cases per model.",
        "Accuracy is in percent; Δ is in percentage points. All values use the same "
        "answer-focused lexical scorer after generation.",
        "Teacher retains the original OLMo response instruction; the selected-KV "
        "and Cartridge controls use the original question format.",
        "",
        "| Model | n | " + " | ".join(LABELS)
        + " | vs Teacher: new right / new wrong | Δ pp | vs Selected: new right / new wrong | Δ pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, record in result["models"].items():
        rows = record["rows"]
        _, gt, lt, dt = _pair(rows, "teacher")
        _, gz, lz, dz = _pair(rows, "zero")
        rates = " | ".join(_pct(_rate(rows, arm)) for arm in ARMS)
        lines.append(
            f"| {model} | {len(rows)} | {rates} | {gt} / {lt} | {_pct(dt)} "
            f"| {gz} / {lz} | {_pct(dz)} |"
        )
    for model, record in result["models"].items():
        rows = record["rows"]
        lines += ["", f"## {model}: by task", "", *_table(rows, "task", _task_key)]
        lines += ["", f"## {model}: by context length", "", *_table(rows, "length", _length_key)]
        n, gain, loss, delta = _pair(rows, "old_cartridge")
        lines += [
            "",
            f"Strictly matched original-SD comparison: {n} cases; "
            f"new right / new wrong = {gain} / {loss}; Δ = {_pct(delta)} pp.",
        ]
    lines += [
        "",
        "A partial run summarizes completed cases only. A matched old-SD comparison "
        "uses only cases that passed the selection, teacher, zero-output, and proposal checks.",
        "The reported advantage against Teacher is a method-level comparison; "
        "the same-format selected-KV control isolates the effect of the trained reading state.",
    ]
    if result["mode"] == "holdout":
        lines.append(
            "These descriptive tables include all holdout cases; the predeclared "
            "acceptance decision also excludes previously inspected source examples."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved latest.json with scored binary rows")
    parser.add_argument("output", type=Path, help="Report Markdown path")
    args = parser.parse_args()
    text = render(json.loads(args.input.read_text()))
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
