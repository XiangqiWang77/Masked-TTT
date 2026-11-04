#!/usr/bin/env python3
"""Replay saved GPT proposal previews; use no model, API or evaluation labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.truncated_study_json.parser import original_parse, parse_with_truncated_recovery


def collect(workspace):
    run = workspace / "query_cartridges_cross_v4_20261103/runs/gptoss20b-babilong"
    split = workspace / "query_cartridges_gpt_v4_holdout_20261103/split.json"
    protocol = json.loads((run / "protocol.json").read_text())
    entries = json.loads(Path(protocol["index"]).read_text())
    index = {entry["key"]: entry for entry in entries}
    cases = json.loads(split.read_text())["development"]
    if len(index) != len(entries) or len(cases) != 200 or len(set(cases)) != 200:
        raise ValueError("Expected the frozen 200-case development set")
    counts = Counter()
    with open(protocol["source_cases"], "rb") as source:
        for key in cases:
            saved = json.loads((run / "cases" / key / "masked_ttt.json").read_text())
            entry = index[key]
            source.seek(entry["offset"])
            case = json.loads(source.read(entry["bytes"]))
            if (saved["key"] != key or case["key"] != key or
                saved["selection"]["reference_answers_used"] is not False or
                set(case) & {"answer", "answers", "references", "gold", "answer_spans"}):
                raise ValueError("Unexpected labeled input or mismatched case")
            scored = json.loads(Path(saved["selection"]["score_file"]).read_text())
            spans = [tuple(span) for span in scored["source_spans"]]
            selected = sorted({i for group in saved["selection"]["selected_original_chunks"] for i in group})
            preview = saved["self_study"].get("generated_preview", "")
            if not isinstance(preview, str) or len(preview) > 512:
                raise ValueError("Unexpected saved preview")
            args = (preview, case["document"], spans, selected, case["query"])
            before, old_rejections = original_parse(*args, limit=2)
            after, new_rejections = parse_with_truncated_recovery(*args, limit=2)
            identity = lambda item: (item.question, item.answer, item.evidence, item.source_chunk)
            previous = {identity(item) for item in before}
            current = {identity(item) for item in after}
            if not previous.issubset(current):
                raise ValueError("Recovery displaced an originally accepted proposal")
            counts["original_proposals"] += len(before)
            counts["recovery_proposals"] += len(after)
            counts["new_proposals"] += len(current - previous)
            counts["cases_with_new_proposals"] += bool(current - previous)
            counts["invalid_json_before"] += bool(old_rejections.get("invalid_json"))
            counts["invalid_json_after"] += bool(new_rejections.get("invalid_json"))
            counts["preview_at_512_characters"] += len(preview) == 512
    return dict(model="gpt", cases=len(cases), source_questions=len({key.split("-", 2)[2] for key in cases}),
                counts=dict(counts), preview_character_limit=512,
                saved_preview_may_be_shorter_than_original_generation=True,
                independent_verification_performed=False, final_answers_evaluated=False,
                reference_answers_used=False, model_calls=0, online_api_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = collect(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
