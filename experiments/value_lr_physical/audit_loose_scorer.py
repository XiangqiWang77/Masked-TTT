#!/usr/bin/env python3
"""Check the same task-aware lexical scorer on completed BABILONG outputs.

This is a post-generation screening audit, not an LLM semantic judgment. It
exports only aggregate counts and does not modify any prediction or score.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


OBJECTS = re.compile(r"(?i)\b(apple|football|milk)s?\b")
UNCERTAIN_OR_NEGATED = re.compile(
    r"(?i)\b(?:not|never|no|without|unknown|unclear|maybe|perhaps|possibly|unsure|or)\b"
)


def qa8_possible_missed_set(segment, references, correct):
    """Flag a narrow exact-object-set case for human review; never rescore it."""
    expected = {match.group(1).lower() for answer in references for match in OBJECTS.finditer(answer)}
    found = {match.group(1).lower() for match in OBJECTS.finditer(segment)}
    return bool(not correct and expected and found == expected and not UNCERTAIN_OR_NEGATED.search(segment))


def collect(root, references, scorer, model):
    folders = {"qwen": "query_cartridges_qwen_olmo1000_20261104",
               "gpt": "query_cartridges_gpt_olmo1000_20261103"}
    path = root / folders[model]
    report = json.loads((path / "latest.json").read_text())
    if report["metric"] != scorer.BABILONG_SCORER_VERSION or report["completed"] != 1000:
        raise ValueError("A completed 1,000-case report using the same scorer is required")
    rows = report["rows"]
    if len(rows) != 1000 or len({row["case"] for row in rows}) != 1000:
        raise ValueError("The completed report must contain 1,000 unique cases")
    result = {}
    for row in rows:
        case, task = row["case"], row["task"]
        ref = references[case]
        if ref["task"] != task:
            raise ValueError("The reference task differs from the completed report")
        item = json.loads((path / "cases" / case / "weight_0.00.json").read_text())
        control = item["standalone_cartridge"]["paired_prompt_control"]
        if (not control["same_selected_kv"] or not control["same_generation_limit"] or
            control["reference_answers_used"] or control["teacher_answer_in_student_prompt"]):
            raise ValueError("The selected-cache controls are not valid")
        arms = {"whole": (item["self_judge"]["native_prediction"], row["whole"]),
                "zero": (control["zero_original"]["prediction"], row["selective"]),
                "teacher": (item["teacher"]["prediction"], row["teacher"]),
                "cartridge": (control["trained_original"]["prediction"], row["cartridge"])}
        for arm, (prediction, saved_score) in arms.items():
            correct, resolved = scorer.score_prediction(prediction, ref["answers"], "babilong", task)
            if correct != saved_score:
                raise ValueError("The saved answer score differs from the unified scorer")
            totals = result.setdefault(arm, Counter())
            totals["answers_checked"] += 1
            totals["correct"] += int(correct)
            totals["unresolved"] += int(not resolved)
            segment = scorer.focused_segment(prediction, task) if prediction else ""
            if task in ("qa1", "qa2", "qa3", "qa4", "qa5") and correct:
                extracted = scorer.babilong_answer_segment(segment, task)
                totals["correct_with_multiple_answer_labels"] += int(
                    len(scorer.answer_candidates(extracted, task)) > 1)
            if task in ("qa6", "qa9", "qa10"):
                totals["yes_no_maybe_unresolved"] += int(not resolved)
            if task == "qa8":
                totals["qa8_possible_missed_set"] += int(
                    qa8_possible_missed_set(segment, ref["answers"], correct))
    return {arm: dict(counts) for arm, counts in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    import report_hybrid_partial as scorer
    if scorer.BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected unified scorer")
    references = json.loads(args.references.read_text())
    data = dict(metric=scorer.BABILONG_SCORER_VERSION,
        scope="Two 1,000-case completed paired runs; 250 source questions at four lengths per model",
        screening_only=True, semantic_review=False, changes_scores=False, model_or_api_calls=0,
        models={model: collect(args.workspace, references, scorer, model) for model in ("qwen", "gpt")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for model, arms in data["models"].items():
        print(model, json.dumps(arms))


if __name__ == "__main__":
    main()
