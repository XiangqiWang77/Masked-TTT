#!/usr/bin/env python3
"""Audit reproducibility of completed, independently generated paired controls.

Only case IDs are taken from the already-scored snapshot. No reference answer,
score, generated answer, or individual case ID is included in the output.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


SELECTION_FIELDS = ("selected_original_chunks", "selected_chunks", "total_chunks", "selected_tokens")


def pair_flags(candidate, baseline):
    """Return equality indicators; no individual generated text is retained."""
    if candidate["key"] != baseline["key"] or candidate["model_short"] != baseline["model_short"]:
        raise ValueError("Pair does not identify the same case and backbone")
    if not all(isinstance(x["gpu"], str) and x["gpu"] for x in (candidate, baseline)):
        raise ValueError("GPU model must be recorded for both requests")
    csel, bsel = candidate["selection"], baseline["selection"]
    same_selection = all(csel[k] == bsel[k] for k in SELECTION_FIELDS)
    teacher = candidate["teacher"]["tokens"] == baseline["teacher"]["tokens"]
    practice = candidate["self_study"]["checks"] == baseline["self_study"]["checks"]
    zero = (candidate["standalone_cartridge"]["original_question_controls"]["zero"]["tokens"] ==
            baseline["standalone_cartridge"]["original_question_controls"]["zero"]["tokens"])
    return dict(same_gpu=candidate["gpu"] == baseline["gpu"],
                same_node=candidate["node"] == baseline["node"],
                date_recorded=candidate["request_chat_date"] is not None and
                              baseline["request_chat_date"] is not None,
                same_date=candidate["request_chat_date"] == baseline["request_chat_date"],
                selection_different=not same_selection, teacher_tokens_different=not teacher,
                practice_different=not practice, zero_tokens_different=not zero,
                observable_pair_different=not (same_selection and teacher and practice and zero))


def summarize(rows):
    keys = ("same_node", "date_recorded", "same_date", "selection_different",
            "teacher_tokens_different", "practice_different", "zero_tokens_different",
            "observable_pair_different")
    return {"n": len(rows), **{key: sum(row[key] for row in rows) for key in keys}}


def audit(snapshot, candidate_root, baseline_root):
    if snapshot.get("expected_per_model") != 200 or snapshot.get("metric") != "answer_focused_lexical_v2":
        raise ValueError("Unexpected saved paired-development snapshot")
    models = {}
    for model in ("qwen", "gpt"):
        item = snapshot["models"][model]
        cases = [row["case"] for row in item["candidate_rows"]]
        if len(cases) != item["candidate_all"]["n"] or len(set(cases)) != len(cases):
            raise ValueError("Snapshot candidate IDs are incomplete or duplicated")
        rows, gpu_pairs = [], Counter()
        for case in cases:
            paths = [Path(root) / "cases" / model / case / "result.json"
                     for root in (candidate_root, baseline_root)]
            if not all(path.is_file() for path in paths):
                raise ValueError("Snapshot references an unavailable pair")
            c, b = (json.loads(path.read_text()) for path in paths)
            if c["key"] != case or b["key"] != case or c["model_short"] != model:
                raise ValueError("Pair identity does not match the snapshot")
            flags = pair_flags(c, b)
            rows.append(flags)
            gpu_pairs[(b["gpu"], c["gpu"], flags["observable_pair_different"])] += 1
        models[model] = dict(overall=summarize(rows),
            same_gpu=summarize([row for row in rows if row["same_gpu"]]),
            different_gpu=summarize([row for row in rows if not row["same_gpu"]]),
            gpu_pairs=[dict(baseline=old, candidate=new, observable_pair_different=different, n=n)
                       for (old,new,different),n in sorted(gpu_pairs.items())])
    return dict(snapshot_created_utc=snapshot["created_utc"],
        scope="Completed cases named by the saved snapshot, not a live job poll",
        online_api_calls=0, model_calls=0, reference_answers_used=False,
        score_values_used=False, individual_cases_or_generated_text_in_output=False,
        equality_fields=list(SELECTION_FIELDS)+["teacher tokens", "practice verification", "zero-update output tokens"],
        note="Matching output tokens does not establish equality of underlying probability distributions.",
        models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "candidate-root", "baseline-root", "output"):
        parser.add_argument("--"+arg, required=True, type=Path)
    args = parser.parse_args()
    data = audit(json.loads(args.snapshot.read_text()), args.candidate_root, args.baseline_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2)+"\n")
    tmp.replace(args.output)
    for model, values in data["models"].items():
        print(model, "same GPU", values["same_gpu"]["n"], values["same_gpu"]["observable_pair_different"],
              "different GPU", values["different_gpu"]["n"], values["different_gpu"]["observable_pair_different"])


if __name__ == "__main__":
    main()
