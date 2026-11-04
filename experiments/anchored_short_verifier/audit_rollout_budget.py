"""Audit on-policy prefix length from a fixed, already-scored snapshot.

This reads only saved token IDs and scored binary outcomes. It neither changes
the model nor generates or evaluates any answers.
"""
import argparse
import json
from pathlib import Path


def _binary(value):
    if type(value) not in (int, bool) or value not in (0, 1):
        raise ValueError("Expected a binary post-generation outcome")
    return bool(value)


def read_row(saved, result, eos_ids=None):
    training = result["training"]
    first = training["steps"][0]
    if first["question_path"] != "main":
        raise ValueError("The first step must use the original question")
    tokens = first["student_rollout_tokens"]
    limit = training["student_rollout_limit"]
    if type(limit) is not int or limit < 1 or len(tokens) > limit:
        raise ValueError("The student prefix exceeds its configured budget")
    teacher = result["teacher"]
    if type(teacher["complete"]) is not bool:
        raise ValueError("The teacher completion state is required")
    return {
        "case": saved["case"],
        "task": saved["task"],
        "student_prefix_tokens": len(tokens),
        # Reaching the cap alone does not prove that the answer was cut off:
        # a generated EOS can coincide with the configured final position.
        "student_prefix_reaches_cap": len(tokens) == limit,
        "student_cap_without_eos": (
            len(tokens) == limit and tokens[-1] not in eos_ids if eos_ids is not None else None
        ),
        "teacher_generation_complete": teacher["complete"],
        "zero_correct": _binary(saved["selective"]),
        "cartridge_correct": _binary(saved["cartridge"]),
    }


def summarize(rows):
    n = len(rows)
    gain = sum(row["cartridge_correct"] and not row["zero_correct"] for row in rows)
    loss = sum(row["zero_correct"] and not row["cartridge_correct"] for row in rows)
    return {
        "n": n,
        "student_prefix_reaches_cap": sum(row["student_prefix_reaches_cap"] for row in rows),
        "student_cap_without_eos": (
            sum(row["student_cap_without_eos"] for row in rows)
            if rows and all(row["student_cap_without_eos"] is not None for row in rows) else None
        ),
        "teacher_generation_incomplete": sum(not row["teacher_generation_complete"] for row in rows),
        "zero_accuracy": 100 * sum(row["zero_correct"] for row in rows) / n if n else None,
        "cartridge_accuracy": 100 * sum(row["cartridge_correct"] for row in rows) / n if n else None,
        "new_correct": gain,
        "new_wrong": loss,
        "delta_pp": 100 * (gain - loss) / n if n else None,
    }


def audit(snapshot, results_root, eos_ids=None):
    root = Path(results_root)
    rows, skipped = [], []
    for saved in snapshot["rows"]:
        path = root / "cases" / saved["case"] / "weight_0.00.json"
        try:
            result = json.loads(path.read_text())
            rows.append(read_row(saved, result, eos_ids=eos_ids))
        except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            skipped.append({"case": saved["case"], "error": type(exc).__name__})
    groups = {}
    for name, subset in [
        ("all", rows),
        ("at_prefix_cap", [row for row in rows if row["student_prefix_reaches_cap"]]),
        ("shorter_than_prefix_cap", [row for row in rows if not row["student_prefix_reaches_cap"]]),
        ("teacher_incomplete", [row for row in rows if not row["teacher_generation_complete"]]),
    ]:
        groups[name] = summarize(subset)
    return {
        "model": snapshot["model"],
        "snapshot_time": snapshot["updated"],
        "expected_snapshot_cases": len(snapshot["rows"]),
        "groups": groups,
        "rows": rows,
        "unavailable_cases": skipped,
        "new_model_calls": 0,
        "online_api_calls": 0,
        "evaluation_answers_used_in_training": False,
        "eos_ids_supplied": eos_ids is not None,
    }


def render(result):
    lines = [
        "# On-policy prefix budget: fixed-snapshot audit",
        "",
        f"Model: {result['model']}; {result['expected_snapshot_cases']} saved snapshot cases.",
        "",
        "| Group | n | Prefix reaches cap | Cap without EOS | Teacher incomplete | Zero update | Cartridge | New right / wrong | Δ pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    for name, stats in result["groups"].items():
        lines.append(
            f"| {name} | {stats['n']} | {stats['student_prefix_reaches_cap']} | "
            f"{stats['student_cap_without_eos'] if stats['student_cap_without_eos'] is not None else '—'} | "
            f"{stats['teacher_generation_incomplete']} | {fmt(stats['zero_accuracy'])} | "
            f"{fmt(stats['cartridge_accuracy'])} | {stats['new_correct']} / {stats['new_wrong']} | "
            f"{fmt(stats['delta_pp'])} |"
        )
    lines += [
        "",
        f"Unavailable saved results: {len(result['unavailable_cases'])}. Accuracy is percent.",
        "A prefix reaching its budget without EOS was stopped by the prefix cap, but that does "
        "not establish that the final answer was absent, "
        "and group differences do not establish that a longer prefix improves accuracy.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--eos-ids", type=int, nargs="+", default=None,
                        help="All configured model/tokenizer EOS IDs, if known")
    args = parser.parse_args()
    result = audit(json.loads(args.snapshot.read_text()), args.results_root,
                   eos_ids=set(args.eos_ids) if args.eos_ids is not None else None)
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(args.output.with_suffix(".json"))
    temp = args.output.with_suffix(".md.tmp")
    temp.write_text(render(result))
    temp.replace(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
