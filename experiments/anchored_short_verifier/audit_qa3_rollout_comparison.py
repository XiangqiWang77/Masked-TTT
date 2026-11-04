"""Compare saved 8- and 32-token QA3 SD signals on genuinely matched cases.

This reads frozen development IDs and existing outputs, not evaluation labels.
It reports whether a location-shaped token appears after the answer predicate;
it does not determine whether that location is correct.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path


ANSWER_LOCATION = re.compile(
    r"(?i)\b(?:was|were|is|are)\s+(?:located\s+)?(?:in|at|inside)\s+"
    r"(?:the\s+)?(?:bathroom|bedroom|garden|hallway|kitchen|office)\b"
)


def _main_step(result, limit):
    training = result["training"]
    if training["student_rollout_limit"] != limit or training["selected_transfer_weight"] != 0:
        raise ValueError("Unexpected student rollout or transfer objective")
    if not training["steps"] or training["steps"][0]["question_path"] != "main":
        raise ValueError("Missing first original-question update")
    step = training["steps"][0]
    tokens, targets = step["student_rollout_tokens"], step["target_positions"]
    if not tokens or len(tokens) > limit or targets != list(range(len(tokens))):
        raise ValueError("The causal teacher must score every saved rollout position")
    values = (step["teacher_KL_before"], step["gain_gradient_norm"], step["value_gradient_norm"])
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in values) or
        any(x < 0 for x in values[1:])):
        raise ValueError("Invalid logged initial KL or gradient norms")
    return step


def compare_pair(case, old, new, decode):
    old_step, new_step = _main_step(old, 8), _main_step(new, 32)
    old_selection, new_selection = old["selection"], new["selection"]
    old_controls = old["standalone_cartridge"]["paired_prompt_control"]
    new_controls = new["standalone_cartridge"]["original_question_controls"]
    fields = ("question", "answer", "evidence", "source_chunk", "verifier_answer", "accepted")
    proposals = lambda result: [{key: item[key] for key in fields}
                                for item in result["self_study"]["checks"]]
    old_tokens, new_tokens = old_step["student_rollout_tokens"], new_step["student_rollout_tokens"]
    if (old["key"] != case or new["key"] != case or old["task"] != "qa3" or new["task"] != "qa3" or
        old["model"] != new["model"] or old_selection["source_case_key"] != case or
        new_selection["source_case_key"] != case or
        old_selection["selected_original_chunks"] != new_selection["selected_original_chunks"] or
        old["teacher"]["tokens"] != new["teacher"]["tokens"] or
        old_controls["zero_original"]["tokens"] != new_controls["zero"]["tokens"] or
        new_tokens[:len(old_tokens)] != old_tokens or proposals(old) != proposals(new) or
        old_controls["reference_answers_used"] or new_controls["reference_answers_used"] or
        old_controls["teacher_answer_in_student_prompt"] or new_controls["teacher_answer_in_student_prompt"]):
        raise ValueError("Teacher, evidence, zero control, rollout prefix or practices differ")
    return {
        "case": case,
        "old_tokens": len(old_tokens),
        "new_tokens": len(new_tokens),
        "longer": len(new_tokens) > len(old_tokens),
        "old_has_predicated_location": bool(ANSWER_LOCATION.search(decode(old_tokens))),
        "new_has_predicated_location": bool(ANSWER_LOCATION.search(decode(new_tokens))),
        "old_first_kl": old_step["teacher_KL_before"],
        "new_first_kl": new_step["teacher_KL_before"],
        "old_gain_gradient": old_step["gain_gradient_norm"],
        "new_gain_gradient": new_step["gain_gradient_norm"],
        "old_value_gradient": old_step["value_gradient_norm"],
        "new_value_gradient": new_step["value_gradient_norm"],
        "final_cartridge_changed": (
            old_controls["trained_original"]["tokens"] != new_controls["adapted"]["tokens"]
        ),
    }


def audit(case_ids, model, old_root, alternate_root, new_root, decode):
    cases = [case for case in case_ids if "-qa3-" in case]
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate QA3 case")
    rows, missing, rejected = [], [], []
    for case in cases:
        new_file = Path(new_root) / "cases" / model / case / "result.json"
        old_files = [Path(old_root) / "cases" / case / "weight_0.00.json",
                     Path(alternate_root) / "cases" / model / case / "lr_0.0001.json"]
        if not new_file.is_file():
            missing.append({"case": case, "reason": "new_result_unavailable"})
            continue
        new = json.loads(new_file.read_text())
        errors = []
        for old_file in old_files:
            if not old_file.is_file():
                continue
            try:
                row = compare_pair(case, json.loads(old_file.read_text()), new, decode)
                rows.append(row)
                break
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        else:
            if errors:
                rejected.append({"case": case, "reasons": errors})
            else:
                missing.append({"case": case, "reason": "baseline_unavailable"})
    def median(key):
        return statistics.median(row[key] for row in rows) if rows else None
    return {
        "model": model,
        "qa3_expected": len(cases),
        "matched": len(rows),
        "longer": sum(row["longer"] for row in rows),
        "old_with_location": sum(row["old_has_predicated_location"] for row in rows),
        "new_with_location": sum(row["new_has_predicated_location"] for row in rows),
        "old_median_first_kl": median("old_first_kl"),
        "new_median_first_kl": median("new_first_kl"),
        "old_median_gain_gradient": median("old_gain_gradient"),
        "new_median_gain_gradient": median("new_gain_gradient"),
        "old_median_value_gradient": median("old_value_gradient"),
        "new_median_value_gradient": median("new_value_gradient"),
        "final_cartridge_changed": sum(row["final_cartridge_changed"] for row in rows),
        "missing": missing,
        "rejected": rejected,
        "references_read": False,
        "new_model_calls": 0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("model", choices=("qwen", "gpt"))
    parser.add_argument("old_root", type=Path)
    parser.add_argument("alternate_root", type=Path)
    parser.add_argument("new_root", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    result = audit(json.loads(args.manifest.read_text())["case_ids"], args.model, args.old_root,
                   args.alternate_root, args.new_root,
                   lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
