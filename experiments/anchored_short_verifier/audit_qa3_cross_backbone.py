"""Compare old OLMo and Qwen QA3 first steps on a fixed, label-free subset.

Require the same complete source, selected original chunks, and decoded
token-by-token student prefix. Model vocabularies and internal parametrization
still differ, so this is a descriptive signal audit, not a causal estimate.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


SIGNALS = ("teacher_KL_before", "gain_gradient_norm", "value_gradient_norm")


def pair(case, qwen, olmo, qwen_decode, olmo_decode):
    if (qwen["key"] != case or olmo["key"] != case or qwen["task"] != "qa3" or olmo["task"] != "qa3" or
        qwen["model"] != "Qwen/Qwen3-4B-Instruct-2507" or
        olmo["model"] != "allenai/Olmo-3-7B-Instruct"):
        raise ValueError("Unexpected case or task")
    first = []
    for result in (qwen, olmo):
        training = result["training"]
        if (training["objective"] != "on-policy KL(teacher || student)" or
            training["student_rollout_limit"] != 8 or not training["steps"] or
            training["steps"][0]["question_path"] != "main" or
            not result["teacher"]["response_instruction_applied"]):
            raise ValueError("The original main-question protocol differs")
        step = training["steps"][0]
        if step["target_positions"] != list(range(len(step["student_rollout_tokens"]))):
            raise ValueError("The first student prefix was not completely scored")
        if any(type(step[key]) not in (int, float) or not math.isfinite(step[key]) for key in SIGNALS):
            raise ValueError("Invalid first-step signal")
        first.append(step)
    qsel, osel = qwen["selection"], olmo["selection"]
    if qsel["source_case_key"] != case or osel["source_case_key"] != case:
        raise ValueError("The cached score source does not match the case")
    if not qsel["complete_original_document"] or not osel["complete_original_document"]:
        raise ValueError("One model used a compressed working document")
    if qsel["selected_original_chunks"] != osel["selected_original_chunks"]:
        raise ValueError("The selected original chunks differ")
    qt, ot = first[0]["student_rollout_tokens"], first[1]["student_rollout_tokens"]
    if len(qt) != 8 or len(ot) != 8:
        raise ValueError("Both student prefixes must use the eight-token cap")
    if [qwen_decode([token]) for token in qt] != [olmo_decode([token]) for token in ot]:
        raise ValueError("The token-by-token decoded prefixes differ")
    return {"case": case,
            "qwen": {key: first[0][key] for key in SIGNALS},
            "olmo": {key: first[1][key] for key in SIGNALS}}


def audit(case_ids, qwen_root, olmo_root, qwen_decode, olmo_decode):
    cases = [case for case in case_ids if "-qa3-" in case]
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate QA3 case")
    rows, excluded = [], []
    for case in cases:
        qpath = Path(qwen_root) / "cases" / case / "weight_0.00.json"
        opath = Path(olmo_root) / "cases" / case / "masked_ttt.json"
        try:
            rows.append(pair(case, json.loads(qpath.read_text()), json.loads(opath.read_text()),
                             qwen_decode, olmo_decode))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            excluded.append({"case": case, "reason": str(exc)})
    medians = {
        model: {key: statistics.median(row[model][key] for row in rows) if rows else None
                for key in SIGNALS} for model in ("qwen", "olmo")
    }
    return {"qa3_expected": len(cases), "matched": len(rows),
            "source_questions": len({row["case"].rsplit("-", 1)[-1] for row in rows}),
            "medians": medians, "excluded": excluded, "rows": rows,
            "references_read": False, "new_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("qwen_root", type=Path)
    parser.add_argument("olmo_root", type=Path)
    parser.add_argument("qwen_tokenizer", type=Path)
    parser.add_argument("olmo_tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    qt, ot = Tokenizer.from_file(str(args.qwen_tokenizer)), Tokenizer.from_file(str(args.olmo_tokenizer))
    result = audit(json.loads(args.manifest.read_text())["case_ids"], args.qwen_root, args.olmo_root,
                   lambda ids: qt.decode(ids, skip_special_tokens=True),
                   lambda ids: ot.decode(ids, skip_special_tokens=True))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
