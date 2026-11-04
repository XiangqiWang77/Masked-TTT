"""Audit first-step SD agreement for already-matched cartridge runs.

Only saved model-side signals are read; benchmark answers are not needed.
The first update uses the original question before any self-study path.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def _relative_difference(a, b):
    if type(a) not in (int, float) or type(b) not in (int, float):
        raise ValueError("Expected numerical model-side SD diagnostics")
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Expected finite model-side SD diagnostics")
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def compare_initial_step(original, candidate):
    old = original["training"]["steps"][0]
    new = candidate["training"]["steps"][0]
    if old["question_path"] != "main" or new["question_path"] != "main":
        raise ValueError("The first SD step must use the original question")
    return {
        "same_student_rollout": old["student_rollout_tokens"] == new["student_rollout_tokens"],
        "same_target_positions": old["target_positions"] == new["target_positions"],
        "kl_relative_difference": _relative_difference(old["teacher_KL_before"], new["teacher_KL_before"]),
        "gain_gradient_relative_difference": _relative_difference(
            old["gain_gradient_norm"], new["gain_gradient_norm"]
        ),
        "value_gradient_relative_difference": _relative_difference(
            old["value_gradient_norm"], new["value_gradient_norm"]
        ),
    }


def _summary(rows):
    comparable = [row for row in rows if row["same_student_rollout"] and row["same_target_positions"]]

    def median(name):
        values = [row[name] for row in comparable]
        return statistics.median(values) if values else None

    return {
        "n": len(rows),
        "same_first_rollout_and_positions": len(comparable),
        "median_kl_relative_difference": median("kl_relative_difference"),
        "median_gain_gradient_relative_difference": median("gain_gradient_relative_difference"),
        "median_value_gradient_relative_difference": median("value_gradient_relative_difference"),
        "cases_kl_relative_difference_over_1_percent": sum(row["kl_relative_difference"] > .01 for row in comparable),
        "cases_gain_gradient_relative_difference_over_1_percent": sum(
            row["gain_gradient_relative_difference"] > .01 for row in comparable
        ),
        "cases_value_gradient_relative_difference_over_1_percent": sum(
            row["value_gradient_relative_difference"] > .01 for row in comparable
        ),
    }


def audit(scored, candidate_root, baselines, lr_root):
    candidate_root, lr_root = Path(candidate_root), Path(lr_root)
    groups, skipped = {}, []
    for model, result in scored["models"].items():
        groups[model] = []
        for row in result["rows"]:
            source = row.get("baseline_source")
            if source is None:
                continue
            key = row["case"]
            if source == "full_1000":
                old_path = Path(baselines[model]) / "cases" / key / "weight_0.00.json"
            elif source == "value_lr_development":
                old_path = lr_root / "cases" / model / key / "lr_0.0001.json"
            else:
                raise ValueError("Unexpected matched-baseline source")
            new_path = candidate_root / "cases" / model / key / "result.json"
            try:
                old = json.loads(old_path.read_text())
                new = json.loads(new_path.read_text())
                signal = compare_initial_step(old, new)
            except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                skipped.append({"model": model, "case": key, "error": type(exc).__name__})
                continue
            groups[model].append({"case": key, "baseline_source": source, **signal})
    return {
        "models": {model: {"summary": _summary(rows), "rows": rows} for model, rows in groups.items()},
        "unavailable_cases": skipped,
        "reference_answers_used": False,
        "online_api_calls": 0,
    }


def render(result):
    lines = [
        "# Same-case first-step SD audit",
        "",
        "The first update uses the original question. These comparisons do not use reference answers.",
        "",
        "| Model | Matched steps | Same rollout and positions | Median relative KL difference "
        "| Median relative gain-gradient difference | Median relative value-gradient difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda x: "—" if x is None else f"{100 * x:.3f}%"
    for model, record in result["models"].items():
        s = record["summary"]
        lines.append(
            f"| {model} | {s['n']} | {s['same_first_rollout_and_positions']} | "
            f"{fmt(s['median_kl_relative_difference'])} | {fmt(s['median_gain_gradient_relative_difference'])} "
            f"| {fmt(s['median_value_gradient_relative_difference'])} |"
        )
    lines += [
        "",
        f"Unreadable or incomplete matched pairs: {len(result['unavailable_cases'])}.",
        "Medians and threshold counts include only identical rollouts and target positions. "
        "Differences can arise from numerical execution as well as an implementation change; "
        "they are a reproduction diagnostic, not an accuracy metric.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("qwen_baseline", type=Path)
    parser.add_argument("gpt_baseline", type=Path)
    parser.add_argument("lr_baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.scored.read_text()), args.candidate_root,
                   {"qwen": args.qwen_baseline, "gpt": args.gpt_baseline}, args.lr_baseline)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output.with_suffix(".json"))
    temporary = args.output.with_suffix(".md.tmp")
    temporary.write_text(render(result))
    temporary.replace(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
