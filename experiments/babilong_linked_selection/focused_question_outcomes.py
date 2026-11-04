#!/usr/bin/env python3
"""Join exploratory support gains with saved baseline answer outcomes."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection import question_overlap_dev as base
from experiments.babilong_linked_selection.focused_question_selector import focused_question_overlap


def summarize(rows):
    if any(r["coverage"][name] not in ("complete", "incomplete") for r in rows
           for name in ("proper_where", "lexical")):
        raise ValueError("Ambiguous or unavailable support coverage cannot be compared")
    newly = [r for r in rows if r["coverage"]["proper_where"] == "incomplete" and
             r["coverage"]["lexical"] == "complete"]
    if any(r["selection_changed"]["proper_where"] for r in newly):
        raise ValueError("Saved baseline answers do not match the prior selection on a newly covered case")
    lost = [r for r in rows if r["coverage"]["proper_where"] == "complete" and
            r["coverage"]["lexical"] == "incomplete"]
    return dict(cases=len(rows), newly_complete=len(newly), lost_complete=len(lost),
                newly_complete_source_questions=len({r["source"] for r in newly}),
                newly_complete_old_teacher_wrong=sum(not r["teacher"] for r in newly),
                newly_complete_old_full_correct_teacher_wrong=sum(r["whole"] and not r["teacher"] for r in newly),
                newly_complete_old_teacher_correct=sum(r["teacher"] for r in newly))


def collect(workspace):
    old_select, old_summarize = base.question_overlap, base.summarize
    captured = []
    def candidate(document, question, spans, records, fraction, *, mode):
        if mode == "lexical":
            return focused_question_overlap(document, question, spans, records, fraction, include_give=True)
        return old_select(document, question, spans, records, fraction, mode=mode)
    def observe(rows, name):
        if name == "lexical" and len(rows) == 200:
            captured.append(summarize(rows))
        return old_summarize(rows, name)
    base.question_overlap, base.summarize = candidate, observe
    try:
        output = base.collect(Path(workspace))
    finally:
        base.question_overlap, base.summarize = old_select, old_summarize
    if output["cases"] != 200 or output["source_questions"] != 50 or len(captured) != 1:
        raise ValueError("Expected one complete development cohort")
    return dict(summary=captured[0], selection_before_annotations=True,
                selection_uses_labels=False, answers_only_used_post_generation=True,
                candidate_answers_not_generated=True, online_api_calls=0, new_model_calls=0,
                selector_designed_after_reviewing_development_outcomes=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
