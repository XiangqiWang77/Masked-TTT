"""Summarize a completed original-selection run without rerunning the model.

All scores come from the same frozen, post-generation report.  The whole-view
and selected-view controls have the same question format within each pair.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


ARMS = ("whole", "whole_adapted", "selective", "teacher", "cartridge")


def summarize(rows):
    if any(type(row[key]) is not int or row[key] not in (0, 1)
           for row in rows for key in ARMS):
        raise ValueError("Each method needs a binary score")
    n = len(rows)
    if not n:
        raise ValueError("No scored rows")

    def pair(after, before):
        gains = sum(row[after] > row[before] for row in rows)
        losses = sum(row[after] < row[before] for row in rows)
        return dict(new_correct=gains, new_wrong=losses,
                    delta_pp=100 * (gains - losses) / n)

    whole_only = [row for row in rows if row["whole"] and not row["selective"]]
    selected_only = [row for row in rows if row["selective"] and not row["whole"]]
    full_gains = [row for row in rows if row["whole_adapted"] and not row["whole"]]
    outcomes = Counter((row["selective"], row["cartridge"]) for row in full_gains)
    return dict(
        n=n,
        source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
        accuracy={key: 100 * sum(row[key] for row in rows) / n for key in ARMS},
        full_update=pair("whole_adapted", "whole"),
        cartridge_update=pair("cartridge", "selective"),
        cartridge_vs_teacher=pair("cartridge", "teacher"),
        whole_only=dict(n=len(whole_only),
            adapted_full_correct=sum(row["whole_adapted"] for row in whole_only),
            cartridge_correct=sum(row["cartridge"] for row in whole_only)),
        whole_only_evidence_coverage=(dict(sorted(Counter(row["coverage"] for row in whole_only).items()))
                                     if all("coverage" in row for row in rows) else None),
        selected_only=dict(n=len(selected_only),
            cartridge_correct=sum(row["cartridge"] for row in selected_only)),
        full_update_new_correct_selected_outcomes=dict(
            n=len(full_gains),
            selected_already_correct_retained=outcomes[(1, 1)],
            selected_already_correct_lost=outcomes[(1, 0)],
            selected_wrong_corrected=outcomes[(0, 1)],
            selected_wrong_still_wrong=outcomes[(0, 0)],
        ),
    )


def collect(report):
    if (report["dataset"] != "BABILONG" or report["model"] != "Qwen/Qwen3-4B-Instruct-2507"
            or report["metric"] != "answer_focused_lexical_v2"
            or report["expected"] != 1000 or report["completed"] != 1000
            or report["uses_extra_transfer_loss"] is not False
            or report["reference_answers_used_only_posthoc"] is not True
            or report["online_API_calls"] != 0 or report["errors"]):
        raise ValueError("Expected the completed, original-method Qwen run")
    rows = report["rows"]
    if len(rows) != 1000 or len({row["case"] for row in rows}) != 1000:
        raise ValueError("Expected 1,000 distinct scored cases")
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    overall = summarize(rows)
    if overall["source_questions"] != 250:
        raise ValueError("Expected 250 sources, each evaluated at four lengths")
    return dict(model=report["model"], dataset=report["dataset"], metric=report["metric"],
                selection=report["selection"], source_updated=report["updated"],
                reference_answers_used_only_posthoc=True, new_model_calls=0,
                online_api_calls=0, overall=overall,
                groups={task: summarize(items) for task, items in sorted(groups.items())})


def render(result):
    overall = result["overall"]
    lines = ["# Qwen 1,000-question whole-to-cartridge audit", "",
             "| Group | n | Whole | Adapted whole | Selected zero | ICL Teacher | Cartridge |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, entry in [("All", overall), *sorted(result["groups"].items(),
                                  key=lambda x: int(x[0][2:]))]:
        a = entry["accuracy"]
        lines.append(f"| {label} | {entry['n']} | {a['whole']:.2f}% | "
                     f"{a['whole_adapted']:.2f}% | {a['selective']:.2f}% | "
                     f"{a['teacher']:.2f}% | {a['cartridge']:.2f}% |")
    full = overall["full_update"]
    cartridge = overall["cartridge_update"]
    whole_only = overall["whole_only"]
    selected_only = overall["selected_only"]
    gains = overall["full_update_new_correct_selected_outcomes"]
    coverage = overall["whole_only_evidence_coverage"]
    lines += ["", "## Paired changes", "",
        f"* Full Student: {full['new_correct']} new correct, {full['new_wrong']} new wrong "
        f"({full['delta_pp']:+.2f} percentage points).",
        f"* Standalone Cartridge against zero-update selected KV: {cartridge['new_correct']} "
        f"new correct, {cartridge['new_wrong']} new wrong ({cartridge['delta_pp']:+.2f} points).",
        f"* Whole correct / selected zero wrong: {whole_only['n']} cases; the adapted whole "
        f"is correct on {whole_only['adapted_full_correct']}, and the Cartridge on "
        f"{whole_only['cartridge_correct']}.",
        *([f"* In those whole-only cases, annotated support was incomplete in "
           f"{coverage.get('incomplete', 0)} and complete in {coverage.get('complete', 0)}; "
           "support annotations are used only after generation."] if coverage is not None else []),
        f"* Selected zero correct / whole wrong: {selected_only['n']} cases; "
        f"the Cartridge remains correct on {selected_only['cartridge_correct']}.",
        f"* Of {gains['n']} new correct answers on the full path, selected zero was already "
        f"correct and remained correct on {gains['selected_already_correct_retained']}; "
        f"selected zero was wrong on {gains['selected_wrong_corrected'] + gains['selected_wrong_still_wrong']}, "
        f"of which the Cartridge corrected {gains['selected_wrong_corrected']}.", "",
        "These groups are defined using post-generation answer labels. They describe paired "
        "outcomes and do not by themselves establish the causal reason for an answer change. "
        "The 1,000 rows represent 250 source questions at four context lengths.", "",
        "Whole, adapted whole, selected zero, and Cartridge use the original question format; "
        "the ICL Teacher additionally has the original generic response instruction. This format "
        "difference matters for the Teacher comparison. The score is the same task-aware "
        "loose lexical scorer for every arm, rather than an LLM semantic judge.", "",
        "The report uses the completed original Decision top-10% selection, with no model or "
        "online API call. Adaptation uses selected ICL Teacher, full Student on-policy KL, "
        "and independent selected-cache materialization; no reverse transfer loss is added.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = collect(json.loads(args.snapshot.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(render(result))
    print(render(result))


if __name__ == "__main__":
    main()
