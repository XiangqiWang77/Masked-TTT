"""Check whether a stronger QA3 SD signal changes full or cartridge answers.

Inputs are already-generated 8- and 32-token results. References are read only
here, after inference, to score predictions with the common BABILONG metric.
"""
import argparse
import json
from pathlib import Path
import sys

from experiments.anchored_short_verifier.audit_qa3_rollout_comparison import compare_pair


def summarize(rows):
    keys = ("native_old", "adapted_old", "native_new", "adapted_new", "zero", "cartridge", "teacher")
    changes = ("old_full_changed", "new_full_changed", "full_8_32_changed",
               "selected_8_32_changed", "selected_update_changed")
    if any(type(row[key]) is not int or row[key] not in (0, 1) for row in rows for key in keys):
        raise ValueError("Answer scores must be binary")
    if any(type(row[key]) is not bool for row in rows for key in changes):
        raise ValueError("Output changes must be boolean")
    pairs = {}
    for old, new in (("native_new", "adapted_new"), ("adapted_old", "adapted_new"),
                     ("zero", "cartridge"), ("teacher", "cartridge")):
        pairs[old + "_to_" + new] = dict(
            new_correct=sum(row[new] and not row[old] for row in rows),
            new_wrong=sum(row[old] and not row[new] for row in rows))
    return dict(n=len(rows), source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
                correct={key: sum(row[key] for row in rows) for key in keys},
                output_changes={key: sum(row[key] for row in rows) for key in changes}, pairs=pairs)


def collect(audit, old_root, new_root, references, score, decode):
    if audit["model"] != "qwen" or audit["references_read"] or audit["new_model_calls"] != 0:
        raise ValueError("Need the fixed reference-free Qwen QA3 rollout audit")
    rows = []
    for item in audit["rows"]:
        case = item["case"]
        old = json.loads((Path(old_root) / "cases" / case / "weight_0.00.json").read_text())
        new = json.loads((Path(new_root) / "cases/qwen" / case / "result.json").read_text())
        if compare_pair(case, old, new, decode) != item:
            raise ValueError("The saved strict-pair audit does not match its source result")
        ref = references[case]
        if ref["task"] != "qa3":
            raise ValueError("Unexpected question type")
        grade = lambda text: int(score(text, ref["answers"], "babilong", "qa3")[0])
        a, b = old["self_judge"], new["self_judge"]
        old_control = old["standalone_cartridge"]["paired_prompt_control"]
        new_control = new["standalone_cartridge"]["original_question_controls"]
        rows.append(dict(case=case,
            native_old=grade(a["native_prediction"]), adapted_old=grade(a["adapted_prediction"]),
            native_new=grade(b["native_prediction"]), adapted_new=grade(b["adapted_prediction"]),
            zero=grade(new_control["zero"]["prediction"]),
            cartridge=grade(new_control["adapted"]["prediction"]), teacher=grade(new["teacher"]["prediction"]),
            old_full_changed=a["native_prediction"] != a["adapted_prediction"],
            new_full_changed=b["native_prediction"] != b["adapted_prediction"],
            full_8_32_changed=a["adapted_prediction"] != b["adapted_prediction"],
            selected_8_32_changed=old_control["trained_original"]["tokens"] != new_control["adapted"]["tokens"],
            selected_update_changed=new_control["zero"]["tokens"] != new_control["adapted"]["tokens"]))
    if len(rows) != audit["matched"] or len({row["case"] for row in rows}) != len(rows):
        raise ValueError("Matched case counts differ")
    return dict(metric="answer_focused_lexical_v2", model="qwen", overall=summarize(rows),
                evaluation_answers_used_only_posthoc=True, new_model_calls=0, online_api_calls=0, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("audit", "old-root", "new-root", "references", "scorer-dir", "tokenizer", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    from tokenizers import Tokenizer
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected shared metric")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    result = collect(json.loads(args.audit.read_text()), args.old_root, args.new_root,
                     json.loads(args.references.read_text()), score_prediction,
                     lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True))
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
