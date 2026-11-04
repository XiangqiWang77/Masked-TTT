#!/usr/bin/env python3
"""Audit source-verified self-study against completed Qwen paired outcomes.

This is descriptive, post-generation analysis; the output contains no source
text, reference answers, predictions, or case IDs. It never calls a model.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


ARMS = ("whole", "teacher", "zero", "cartridge", "zero_teacher_format", "cartridge_teacher_format")
PAIRS = (("zero", "cartridge"), ("zero_teacher_format", "cartridge_teacher_format"),
         ("teacher", "cartridge"))


def summarize(rows):
    for row in rows:
        if (type(row["verified"]) is not int or row["verified"] not in (0, 1, 2) or
            any(type(row.get(arm)) is not int or row[arm] not in (0, 1) for arm in ARMS)):
            raise ValueError("Expected verified counts and paired binary outcomes")
    n = len(rows)
    pairs = {}
    for base, candidate in PAIRS:
        good = sum(not row[base] and row[candidate] for row in rows)
        bad = sum(row[base] and not row[candidate] for row in rows)
        pairs[base] = dict(new_correct=good, new_wrong=bad,
                           delta_pp=100 * (good - bad) / n if n else None)
    rejections = Counter()
    for row in rows:
        rejections.update(row["rejections"])
        if any(type(row[name]) is not bool for name in ("original_tokens_changed", "teacher_format_tokens_changed")):
            raise ValueError("The generated-token pairing is incomplete")
    return dict(n=n, source_questions=len({row["source"] for row in rows}),
                verified_at_least_one=sum(row["verified"] > 0 for row in rows),
                accuracy={arm: 100 * sum(row[arm] for row in rows) / n if n else None for arm in ARMS},
                pairs=pairs, rejection_counts=dict(sorted(rejections.items())),
                original_tokens_changed=sum(row["original_tokens_changed"] for row in rows),
                teacher_format_tokens_changed=sum(row["teacher_format_tokens_changed"] for row in rows))


def collect(workspace, snapshot):
    dev = workspace / "query_cartridges_linked_selection_20261104"
    held = workspace / "query_cartridges_linked_holdout_20261104"
    manifests = [json.loads((root / "manifest.json").read_text()) for root in (dev, held)]
    dev_cases, held_cases = (manifest["case_ids"] for manifest in manifests)
    if (len(dev_cases) != 200 or len(set(dev_cases)) != 200 or len(held_cases) != 800 or
        len(set(held_cases)) != 800 or set(dev_cases) & set(held_cases) or
        set(manifests[0]["holdout_case_ids"]) != set(held_cases) or
        snapshot["metric"] != "answer_focused_lexical_v2" or snapshot["selected_fraction"] != .1 or
        snapshot["selection_policy"] != "decision_entity_link_v1" or
        snapshot["models"]["qwen"]["state"] != "complete" or
        snapshot["models"]["qwen"]["combined"]["n"] != 1000):
        raise ValueError("Expected the frozen, completed Qwen 1,000-case protocol")
    protocols = [json.loads(Path(m["model_protocols"]["qwen"]).read_text()) for m in manifests]
    if protocols[0] != protocols[1]:
        raise ValueError("Inference protocols differ")
    references = json.loads((workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json").read_text())
    sys.path.insert(0, str(workspace / "monitor"))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected scorer")
    rows = []
    for split, root, cases in (("development", dev, dev_cases), ("holdout", held, held_cases)):
        for key in cases:
            result = json.loads((root / "cases/qwen" / key / "result.json").read_text())
            reference = references[key]
            selection, study, training = result["selection"], result["self_study"], result["training"]
            pair = result["standalone_cartridge"]["paired_prompt_control"]
            count = study["verified"]
            if (result["key"] != key or result["task"] != reference["task"] or
                result["model"] != "Qwen/Qwen3-4B-Instruct-2507" or
                selection["source_case_key"] != key or not selection["complete_original_document"] or
                selection["selection_policy"] != "decision_entity_link_v1" or
                selection["reference_answers_used"] or selection["online_api_calls"] != 0 or
                study["reference_answers_used"] or study["student_answer_hint_used"] or
                type(count) is not int or count not in (0, 1, 2) or
                training["objective"] != "on-policy KL(teacher || student)" or
                training["selected_transfer_weight"] != 0 or
                training["accepted_steps"] != len(training["steps"]) or
                len(training["steps"]) != 2 + count or
                not training["teacher_conditioned_on_student_prefix"] or
                not training["student_rollouts_refreshed_each_step"] or
                not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
                result["teacher_answer_in_final_prompt"] or
                not pair["same_selected_kv"] or not pair["same_backbone_and_device"] or
                not pair["same_generation_limit"] or pair["control_update_steps"] != 0 or
                pair["teacher_answer_in_student_prompt"] or pair["reference_answers_used"]):
                raise ValueError("The saved training or paired-answer protocol changed")
            predictions = dict(whole=result["self_judge"]["native_prediction"],
                teacher=result["teacher"]["prediction"], zero=pair["zero_original"]["prediction"],
                cartridge=pair["trained_original"]["prediction"],
                zero_teacher_format=pair["zero_teacher_suffix"]["prediction"],
                cartridge_teacher_format=pair["trained_teacher_suffix"]["prediction"])
            scores = {}
            for arm, prediction in predictions.items():
                value = score_prediction(prediction, reference["answers"], "babilong", reference["task"])[0]
                if type(value) not in (int, float) or value not in (0, 1):
                    raise ValueError("Expected the common binary score")
                scores[arm] = int(value)
            rejections = study["rejections"]
            if not isinstance(rejections, dict) or any(type(v) is not int or v < 0 for v in rejections.values()):
                raise ValueError("Invalid exercise rejection counts")
            token_pairs = [(pair["zero_" + suffix]["tokens"], pair["trained_" + suffix]["tokens"])
                           for suffix in ("original", "teacher_suffix")]
            if any(not isinstance(tokens, list) for p in token_pairs for tokens in p):
                raise ValueError("Missing paired token sequences")
            rows.append(dict(source=key.split("-", 2)[2], task=reference["task"], split=split,
                             verified=count, rejections=rejections,
                             original_tokens_changed=token_pairs[0][0] != token_pairs[0][1],
                             teacher_format_tokens_changed=token_pairs[1][0] != token_pairs[1][1], **scores))
    if len(rows) != 1000 or len({row["source"] for row in rows}) != 250:
        raise ValueError("Missing completed cases or source questions")
    overall = summarize(rows)
    frozen = snapshot["models"]["qwen"]["combined"]["accuracy"]
    if any(not math.isclose(overall["accuracy"][arm], frozen[arm], abs_tol=1e-9) for arm in ARMS):
        raise ValueError("Rescored results differ from the frozen report")
    subset = lambda pred: summarize([row for row in rows if pred(row)])
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), model="Qwen3-4B-Instruct-2507",
        metric="answer_focused_lexical_v2", no_new_model_calls=True, online_api_calls=0,
        references_used_only_for_post_generation_scoring=True, development_used_for_selection=True,
        source_disjoint_holdout=True, source_questions_repeat_at_four_lengths=True,
        teacher_has_extra_generic_format_instruction=True, overall=overall,
        by_verified={str(i): subset(lambda row: row["verified"] == i) for i in (0, 1, 2)},
        verified_any=subset(lambda row: row["verified"] > 0),
        by_task={task: subset(lambda row: row["task"] == task) for task in sorted({row["task"] for row in rows})},
        by_split={split: subset(lambda row: row["split"] == split) for split in ("development", "holdout")},
        by_split_verified={split: {label: subset(lambda row: row["split"] == split and
                                                    (row["verified"] > 0) == present)
                                   for label, present in (("none", False), ("at_least_one", True))}
                           for split in ("development", "holdout")},
        full_only=subset(lambda row: row["whole"] and not row["teacher"]),
        full_only_by_verified={label: subset(lambda row: row["whole"] and not row["teacher"] and
                                                (row["verified"] > 0) == present)
                               for label, present in (("none", False), ("at_least_one", True))},
        by_task_verified={task: {str(i): subset(lambda row: row["task"] == task and row["verified"] == i)
                                 for i in (0, 1, 2)} for task in sorted({row["task"] for row in rows})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.workspace.resolve(), json.loads(args.snapshot.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"overall": data["overall"], "by_verified": data["by_verified"]}, indent=2))


if __name__ == "__main__":
    main()
