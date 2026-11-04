"""Analyze fixed paired answers by the change in selected support coverage.

Decision scores and predictions are already saved. Support labels and answer
references are read only here, after inference; no model or API is called.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


def coverage_group(before, after):
    if before not in ("complete", "incomplete") or after not in ("complete", "incomplete"):
        return "ambiguous"
    if before == "incomplete" and after == "complete":
        return "newly_complete"
    if before == "complete" and after == "incomplete":
        return "lost_complete"
    return "both_complete" if before == "complete" else "both_incomplete"


def summarize(rows):
    n = len(rows)
    def rate(name):
        return 100 * sum(row[name] for row in rows) / n if n else None
    gains = sum(row["linked"] and not row["score10"] for row in rows)
    losses = sum(row["score10"] and not row["linked"] for row in rows)
    return dict(n=n, score10=rate("score10"), linked=rate("linked"),
                linked_zero=rate("linked_zero"), linked_teacher=rate("linked_teacher"),
                gains=gains, losses=losses, delta_pp=100 * (gains-losses)/n if n else None)


def _selected(result):
    if not result["selection"]["complete_original_document"]:
        raise ValueError("Coverage attribution requires the full original document")
    return sorted({i for group in result["selection"]["selected_original_chunks"] for i in group})


def audit(case_ids, coverage, references, score_root, linked_root, grade):
    annotations = {row["case"]: row for row in coverage["rows"]}
    rows, unavailable = [], []
    for key in case_ids:
        try:
            annotated = annotations[key]
            old = json.loads((Path(score_root)/"cases"/key/"weight_0.00.json").read_text())
            new = json.loads((Path(linked_root)/"cases"/"qwen"/key/"result.json").read_text())
            expected_old = annotated["outcomes"]["score10"]["selected"]
            expected_new = annotated["outcomes"]["entity10"]["selected"]
            if (old["key"] != key or new["key"] != key or old["model"] != new["model"] or
                old["task"] != new["task"] or _selected(old) != expected_old or _selected(new) != expected_new or
                old["selection"]["online_api_calls"] != 0 or new["selection"]["online_api_calls"] != 0 or
                old["selection"]["reference_answers_used"] or new["selection"]["reference_answers_used"] or
                old["training"]["selected_transfer_weight"] != 0 or
                new["training"]["selected_transfer_weight"] != 0):
                raise ValueError("The saved case does not match the annotated pair")
            ref = references[key]
            if ref["task"] != old["task"]:
                raise ValueError("The task reference changed")
            old_pair = old["standalone_cartridge"]["paired_prompt_control"]
            new_pair = new["standalone_cartridge"]["paired_prompt_control"]
            score = lambda text: int(grade(text, ref["answers"], "babilong", ref["task"])[0])
            rows.append(dict(
                case=key, task=ref["task"],
                group=coverage_group(annotated["outcomes"]["score10"]["status"],
                                     annotated["outcomes"]["entity10"]["status"]),
                score10=score(old_pair["trained_original"]["prediction"]),
                linked=score(new_pair["trained_original"]["prediction"]),
                linked_zero=score(new_pair["zero_original"]["prediction"]),
                linked_teacher=score(new["teacher"]["prediction"]),
            ))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            unavailable.append(dict(case=key, error=type(exc).__name__))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    return dict(
        model="Qwen3-4B", expected=len(case_ids), overall=summarize(rows),
        groups={name: summarize(grouped[name]) for name in
                ("newly_complete", "lost_complete", "both_complete", "both_incomplete", "ambiguous")},
        rows=rows, unavailable=unavailable, new_model_calls=0, online_api_calls=0,
        reference_answers_used_in_inference=False,
    )


def render(result):
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    lines = [
        "# Linked selection: coverage and paired outcomes",
        "",
        "Qwen3-4B development cases; labels are used only after generation.",
        "",
        "| Support coverage | n | Score-only Cartridge | Linked zero | Linked Teacher | Linked Cartridge | New right / wrong | Δ pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in [("all", result["overall"]), *result["groups"].items()]:
        lines.append(f"| {name} | {stats['n']} | {fmt(stats['score10'])} | {fmt(stats['linked_zero'])} | "
                     f"{fmt(stats['linked_teacher'])} | {fmt(stats['linked'])} | "
                     f"{stats['gains']} / {stats['losses']} | {fmt(stats['delta_pp'])} |")
    lines += ["", f"Unavailable or mismatched cases: {len(result['unavailable'])}.",
              "Coverage is label-based and does not establish that the model used the facts. "
              "Groups are observational and were analyzed on development data.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("score_root", type=Path)
    parser.add_argument("linked_root", type=Path)
    parser.add_argument("scorer_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_root))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected scorer")
    result = audit(json.loads(args.manifest.read_text())["case_ids"],
                   json.loads(args.coverage.read_text()), json.loads(args.references.read_text()),
                   args.score_root, args.linked_root, score_prediction)
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(args.output.with_suffix(".json"))
    temp = args.output.with_suffix(".md.tmp")
    temp.write_text(render(result))
    temp.replace(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
