"""Summarize actual selected-value write error and paired answer outcomes."""
import argparse
import json
import math
from pathlib import Path
import statistics

RATES = ("lr_0.0001", "lr_0.0003", "lr_0.0010")


def summarize(rows):
    if not rows or len({row["case"] for row in rows}) != len(rows):
        raise ValueError("Expected unique nonempty paired cases")
    stats = {}
    for rate in RATES:
        for row in rows:
            if any(type(row[key]) is not int or row[key] not in (0, 1)
                   for key in ("zero", "teacher", rate)):
                raise ValueError("Expected binary saved answers")
            for suffix in ("_changed", "_relative_l2", "_error"):
                value = row[rate + suffix]
                if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                    raise ValueError("Expected finite selected-value write diagnostics")
        def pair(old):
            gains = sum(row[rate] and not row[old] for row in rows)
            losses = sum(row[old] and not row[rate] for row in rows)
            return dict(new_correct=gains, new_wrong=losses,
                        delta_pp=100 * (gains-losses)/len(rows))
        stats[rate] = dict(
            answer_accuracy=100 * sum(row[rate] for row in rows)/len(rows),
            median_changed_components_percent=100 * statistics.median(row[rate + "_changed"] for row in rows),
            median_written_relative_l2_percent=100 * statistics.median(row[rate + "_relative_l2"] for row in rows),
            median_squared_write_error_percent=100 * statistics.median(row[rate + "_error"] for row in rows),
            versus_zero=pair("zero"), versus_teacher=pair("teacher"), versus_original_rate=pair(RATES[0]))
    return dict(n=len(rows), source_questions=len({row["case"].split("-", 2)[2] for row in rows}), rates=stats)


def collect(snapshot):
    if (snapshot["metric"] != "answer_focused_lexical_v2" or not snapshot["development_only"] or
        snapshot["evaluation_answers_used_in_training"] or snapshot["online_api_calls"] != 0 or
        not snapshot["same_teacher_student_kl"] or snapshot["selected_transfer_weight"] != 0 or
        snapshot["errors"] or snapshot["invalid"]):
        raise ValueError("Expected a clean paired original-SD development snapshot")
    rows = snapshot["models"]["gpt"]["rows"]
    return dict(model="GPT-OSS-20B", metric=snapshot["metric"], source_snapshot=snapshot["updated"],
        evaluation_answers_used_only_posthoc=True, new_model_calls=0, online_api_calls=0,
        overall=summarize(rows))


def render(result):
    lines = ["# Selected Value write precision: fixed GPT-OSS development snapshot", "",
        f"{result['overall']['n']} paired cases; {result['overall']['source_questions']} source questions.", "",
        "| Value LR | Accuracy | Components changed | Written relative norm | Squared write error | vs zero (new right / wrong) |",
        "|---|---:|---:|---:|---:|---:|"]
    for rate, stats in result["overall"]["rates"].items():
        pair = stats["versus_zero"]
        lines.append(f"| {rate[3:]} | {stats['answer_accuracy']:.2f} | "
            f"{stats['median_changed_components_percent']:.2f}% | {stats['median_written_relative_l2_percent']:.3f}% | "
            f"{stats['median_squared_write_error_percent']:.2f}% | {pair['new_correct']} / {pair['new_wrong']} |")
    lines += ["", "The error is the squared norm of the BF16-written correction minus the intended correction, "
        "divided by the squared norm of the intended correction, measured on sampled selected Value elements. "
        "Each row is a separate training learning rate, so accuracy differences cannot be attributed to "
        "rounding alone. Answers are scored only after generation; no new model or API call was used.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = collect(json.loads(args.snapshot.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, text in ((".json", json.dumps(result, indent=2) + "\n"), (".md", render(result))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
    print(render(result))


if __name__ == "__main__":
    main()
