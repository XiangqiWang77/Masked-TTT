#!/usr/bin/env python3
"""Compare the same BABILONG cases by OLMo working-document coverage."""
import argparse
import csv
import json
from pathlib import Path
import sys


def paired(rows, teacher_key="teacher", answer_key="cartridge"):
    if not rows:
        raise ValueError("A comparison group cannot be empty")
    pairs = [(int(row[teacher_key]), int(row[answer_key])) for row in rows]
    if any(a not in (0, 1) or b not in (0, 1) for a, b in pairs):
        raise ValueError("Answer scores must be binary")
    n = len(pairs)
    gains = sum(a == 0 and b == 1 for a, b in pairs)
    losses = sum(a == 1 and b == 0 for a, b in pairs)
    return dict(n=n, baseline_accuracy=100 * sum(a for a, _ in pairs) / n,
                cartridge_accuracy=100 * sum(b for _, b in pairs) / n,
                new_correct=gains, new_wrong=losses,
                delta_points=100 * (gains - losses) / n)


def complementarity(rows):
    """Report whether training preserves or recovers each view's unique wins."""
    triples = [(int(row["whole"]), int(row["zero"]), int(row["trained"])) for row in rows]
    if any(value not in (0, 1) for triple in triples for value in triple):
        raise ValueError("Answer scores must be binary")
    whole_only = [triple for triple in triples if triple[0] and not triple[1]]
    selected_only = [triple for triple in triples if triple[1] and not triple[0]]
    return dict(n=len(triples), whole_only=len(whole_only),
                whole_only_recovered=sum(row[2] for row in whole_only),
                selected_only=len(selected_only),
                selected_only_retained=sum(row[2] for row in selected_only))


def read_report(path):
    report = json.loads(path.read_text())
    if report["metric"] != "answer_focused_lexical_v2" or report["completed"] != 1000:
        raise ValueError("Expected a completed paired 1,000-case report")
    rows = {row["case"]: row for row in report["rows"]}
    if len(rows) != 1000:
        raise ValueError("Duplicate or missing paired cases")
    return rows


def read_olmo_paired(root, references_path, scorer_dir, cases):
    """Score the independent OLMo same-cache rerun, after both answers exist."""
    sys.path.insert(0, str(scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected BABILONG scorer")
    references = json.loads(references_path.read_text())
    output = {}
    for case in cases:
        result = json.loads((root / "cases" / case / "paired.json").read_text())
        contract, ref = result["contract"], references[case]
        if (result["key"] != case or result["model"] != "allenai/Olmo-3-7B-Instruct" or
            result["reference_answers_used"] or contract["reference_answers_used"] or
            not contract["same_selected_kv"] or not contract["same_question_suffix"] or
            not contract["same_backbone_and_device"] or contract["control_update_steps"] != 0):
            raise ValueError("OLMo selected-KV zero-update pairing changed")
        output[case] = dict(
            zero=int(score_prediction(result["zero_update"], ref["answers"], "babilong", ref["task"])[0]),
            trained=int(score_prediction(result["trained"], ref["answers"], "babilong", ref["task"])[0]))
    if sum(x["zero"] for x in output.values()) != 443 or sum(x["trained"] for x in output.values()) != 445:
        raise ValueError("The independent OLMo paired rerun no longer matches its fixed aggregate")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmo-snapshot", type=Path, required=True)
    parser.add_argument("--qwen-report", type=Path, required=True)
    parser.add_argument("--gpt-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--olmo-paired-root", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--scorer-dir", type=Path)
    args = parser.parse_args()
    with args.olmo_snapshot.open(newline="") as stream:
        olmo = list(csv.DictReader(stream))
    if len(olmo) != 1000 or len({x["case"] for x in olmo}) != 1000:
        raise ValueError("Expected the original 1,000-case OLMo snapshot")
    others = {"qwen": read_report(args.qwen_report), "gpt": read_report(args.gpt_report)}
    if any(set(rows) != {x["case"] for x in olmo} for rows in others.values()):
        raise ValueError("Models must be compared on identical case IDs")
    optional = (args.olmo_paired_root, args.references, args.scorer_dir)
    if any(optional) and not all(optional):
        raise ValueError("The paired OLMo root, reference file and scorer directory must be supplied together")
    olmo_paired = read_olmo_paired(*optional, [x["case"] for x in olmo]) if all(optional) else None
    data = dict(metric="answer_focused_lexical_v2", online_api_calls=0, new_model_calls=0,
                evaluation_answers_used_only_posthoc=True,
                teacher_and_original_cartridge_formats_differ=True,
                default_baseline="selected ICL Teacher with generic format instruction",
                same_format_training_baseline=dict(
                    olmo="independent rerun: selected zero-update with original question",
                    qwen="selected zero-update with teacher-format instruction",
                    gpt="selected zero-update with teacher-format instruction"),
                groups={})
    if olmo_paired is not None:
        data["complementarity"] = {"olmo": complementarity([
            dict(whole=row["whole"], **olmo_paired[row["case"]]) for row in olmo])}
        for model, rows in others.items():
            data["complementarity"][model] = complementarity([
                dict(whole=row["whole"], zero=row["selective"], trained=row["cartridge"])
                for row in rows.values()])
    for coverage, expected in (("complete", 838), ("compacted", 162)):
        group = [row for row in olmo if row["working_document"] == coverage]
        if len(group) != expected:
            raise ValueError("The OLMo working-document coverage snapshot changed")
        values = {"olmo": paired(group)}
        if olmo_paired is not None:
            values["olmo"]["same_format_training"] = paired(
                [olmo_paired[row["case"]] for row in group], "zero", "trained")
        for model, rows in others.items():
            aligned = [rows[row["case"]] for row in group]
            values[model] = paired(aligned)
            values[model]["same_format_training"] = paired(
                aligned, "selective_teacher_format", "cartridge_teacher_format")
        data["groups"][coverage] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["groups"], indent=2))


if __name__ == "__main__":
    main()
