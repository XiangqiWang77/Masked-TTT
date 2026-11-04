#!/usr/bin/env python3
"""Audit saved self-study JSON failures without reading prompts or labels."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(records, max_tokens):
    if max_tokens <= 0:
        raise ValueError("The generation limit must be positive")
    failures = [row for row in records if row["rejections"].get("invalid_json", 0)]
    if any(type(row["generated_tokens"]) is not int or not 0 <= row["generated_tokens"] <= max_tokens
           for row in records):
        raise ValueError("Unexpected generated-token count")
    hist = Counter(row["generated_tokens"] for row in failures)
    rejection_events = Counter()
    for row in records:
        rejection_events.update(row["rejections"])
    return dict(cases=len(records), cases_with_verified_practice=sum(row["verified"] > 0 for row in records),
                invalid_json_cases=len(failures),
                invalid_json_at_token_limit=hist.get(max_tokens, 0),
                invalid_json_below_token_limit=len(failures)-hist.get(max_tokens, 0),
                rejection_event_counts=dict(sorted(rejection_events.items())),
                invalid_json_token_histogram={str(k): v for k, v in sorted(hist.items())})


def collect(root, model, profile):
    manifest = json.loads((root / "manifest.json").read_text())
    cases = manifest["case_ids"]
    if (len(cases) != 200 or len(set(cases)) != 200 or manifest["keep_fraction"] != .1 or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("Unexpected development selection protocol")
    config = json.loads(profile.read_text())
    if config["self_question_count"] != 2 or config["self_question_max_tokens"] != 192:
        raise ValueError("Unexpected OLMo self-study generation settings")
    records = []
    for key in cases:
        result = json.loads((root / "cases" / model / key / "result.json").read_text())
        study = result["self_study"]
        if (result["key"] != key or study["reference_answers_used"] or study["student_answer_hint_used"]):
            raise ValueError("Unexpected self-study record")
        records.append({field: study[field] for field in ("generated_tokens", "verified", "rejections")})
    result = summarize(records, config["self_question_max_tokens"])
    result.update(model=model, source_questions=len({key.split("-", 2)[2] for key in cases}),
                  max_tokens=config["self_question_max_tokens"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    w = args.workspace.resolve()
    qwen = collect(w / "query_cartridges_linked_selection_20261104", "qwen",
                   w / "query_cartridges_value_lr_20261104/olmo_profile.json")
    gpt = collect(w / "query_cartridges_linked_value_gpt_dev_20261104", "gpt",
                  w / "query_cartridges_linked_value_gpt_dev_20261104/runtime/olmo_profile.json")
    output = dict(qwen=qwen, gpt=gpt, no_model_or_api_calls=True,
                  prediction_text_and_reference_labels_not_used=True,
                  generated_text_not_saved_in_source_records=True,
                  token_count_alone_does_not_establish_why_json_failed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
