#!/usr/bin/env python3
"""Compare focused question-word selectors on the fixed development set."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection import question_overlap_dev as base
from experiments.babilong_linked_selection.focused_question_selector import focused_question_overlap


def collect(workspace):
    previous = base.question_overlap
    def candidate(document, question, spans, records, fraction, *, mode):
        if mode == "mix":
            return focused_question_overlap(document, question, spans, records, fraction)
        if mode == "lexical":
            return focused_question_overlap(document, question, spans, records, fraction, include_give=True)
        return previous(document, question, spans, records, fraction, mode=mode)
    base.question_overlap = candidate
    try:
        data = base.collect(Path(workspace))
    finally:
        base.question_overlap = previous
    if data["cases"] != 200 or data["source_questions"] != 50 or data["fraction"] != .1:
        raise ValueError("The fixed development coverage protocol changed")
    groups = {key: dict(entity25=value["entity25"], where=value["proper_where"],
                        where_carry=value["mix"], where_carry_give=value["lexical"])
              for key, value in data["groups"].items()}
    reference = groups["all"]["where"]
    incremental = {name: {field: groups["all"][name][field] - reference[field]
                          for field in ("complete", "changed_selections", "newly_complete", "lost_complete",
                                        "full_only_newly_complete", "teacher_only_lost_complete")}
                   for name in ("where_carry", "where_carry_give")}
    return dict(cases=200, source_questions=50, fraction=.1, groups=groups,
                decisions={name: base.decision(groups["all"][name])
                           for name in ("where_carry", "where_carry_give")},
                increment_vs_where=incremental,
                candidates_designed_after_reviewing_development_outcomes=True,
                support_and_answers_used_only_posthoc=True, selection_accepts_no_labels=True,
                accuracy_of_new_selections_not_measured=True, online_api_calls=0, new_model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(dict(overall=result["groups"]["all"], decisions=result["decisions"]), indent=2))


if __name__ == "__main__":
    main()
