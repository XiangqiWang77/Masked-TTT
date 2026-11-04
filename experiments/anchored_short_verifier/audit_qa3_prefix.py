"""Inspect a fixed set of saved QA3 student rollouts without reading answers.

The final "was in the" pattern shows that a rollout stops before its proposed
location; it does not claim that full-vocabulary KL contains no useful signal.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path


ENDS_AT_LOCATION_SLOT = re.compile(r"(?i)\b(?:was|were|is|are)\s+in\s+the\s*$")


def audit_result(case, saved, decode):
    if saved["key"] != case or saved["task"] != "qa3":
        raise ValueError("Unexpected case or task")
    training = saved["training"]
    steps = training["steps"]
    if not steps or steps[0]["question_path"] != "main":
        raise ValueError("The first update must use the original question")
    step = steps[0]
    tokens = step["student_rollout_tokens"]
    limit = training["student_rollout_limit"]
    targets = step["target_positions"]
    if not tokens or type(limit) is not int or len(tokens) > limit:
        raise ValueError("Invalid student rollout or limit")
    if any(type(i) is not int or i < 0 or i >= len(tokens) for i in targets):
        raise ValueError("A target position falls outside the saved rollout")
    signals = (step["teacher_KL_before"], step["gain_gradient_norm"], step["value_gradient_norm"])
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in signals) or
        any(value < 0 for value in signals[1:])):
        raise ValueError("The first-step signals must be finite and gradient norms nonnegative")
    text = decode(tokens)
    pair = saved.get("standalone_cartridge", {}).get("paired_prompt_control")
    return {
        "case": case,
        "tokens": len(tokens),
        "limit": limit,
        "reached_cap": len(tokens) == limit,
        "ends_at_location_slot": bool(ENDS_AT_LOCATION_SLOT.search(text)),
        "target_positions": len(targets),
        "first_teacher_kl": signals[0],
        "first_gain_gradient_norm": signals[1],
        "first_value_gradient_norm": signals[2],
        "generated_tokens_changed": (
            pair["zero_original"]["tokens"] != pair["trained_original"]["tokens"]
            if pair is not None else None
        ),
    }


def audit(case_ids, result_root, decode):
    cases = [case for case in case_ids if "-qa3-" in case]
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate QA3 case")
    rows, missing = [], []
    for case in cases:
        path = Path(result_root) / "cases" / case / "weight_0.00.json"
        if not path.is_file():
            missing.append(case)
            continue
        saved = json.loads(path.read_text())
        rows.append(audit_result(case, saved, decode))
    return {
        "cases_expected": len(cases),
        "cases_observed": len(rows),
        "source_questions": len({row["case"].rsplit("-", 2)[-1] for row in rows}),
        "prefixes_at_cap": sum(row["reached_cap"] for row in rows),
        "prefixes_ending_at_location_slot": sum(row["ends_at_location_slot"] for row in rows),
        "both": sum(row["reached_cap"] and row["ends_at_location_slot"] for row in rows),
        "median_first_teacher_kl": statistics.median(row["first_teacher_kl"] for row in rows) if rows else None,
        "median_first_gain_gradient_norm": statistics.median(row["first_gain_gradient_norm"] for row in rows) if rows else None,
        "median_first_value_gradient_norm": statistics.median(row["first_value_gradient_norm"] for row in rows) if rows else None,
        "matched_generation_pairs": sum(row["generated_tokens_changed"] is not None for row in rows),
        "matched_generation_changed": sum(row["generated_tokens_changed"] is True for row in rows),
        "missing_cases": missing,
        "references_read": False,
        "new_model_calls": 0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Frozen development case IDs")
    parser.add_argument("results", type=Path, help="Completed baseline result directory")
    parser.add_argument("tokenizer", type=Path, help="Local tokenizer.json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    case_ids = json.loads(args.manifest.read_text())["case_ids"]
    result = audit(case_ids, args.results, lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
