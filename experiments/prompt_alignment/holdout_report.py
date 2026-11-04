#!/usr/bin/env python3
"""Score the preselected aligned-question SD holdout and combine 1,000 cases."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.babilong_linked_selection.long_rollout_report import validate_new


def summarize(rows):
    n = len(rows)
    fields = ("whole", "zero", "teacher", "aligned")
    if any(type(row[key]) is not int or row[key] not in (0, 1) for row in rows for key in fields):
        raise ValueError("Expected binary answer scores")
    def paired(old):
        correct = sum(row["aligned"] and not row[old] for row in rows)
        wrong = sum(row[old] and not row["aligned"] for row in rows)
        return dict(new_correct=correct, new_wrong=wrong,
                    delta_pp=100 * (correct - wrong) / n if n else None)
    return dict(n=n, source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
        accuracy={key: 100 * sum(row[key] for row in rows) / n if n else None for key in fields},
        aligned_vs_zero=paired("zero"), aligned_vs_teacher=paired("teacher"),
        aligned_vs_whole=paired("whole"))


def validate_candidate(result, case, model, task):
    validate_new(case, result, model, task)
    train = result["training"]
    if (not train.get("student_question_matches_teacher") or train["gain_head_mode"] != "kv" or
        len(train["steps"]) != 2 + result["self_study"]["verified"] or
        result["standalone_cartridge"].get("answer_format") != "original"):
        raise ValueError("Aligned-question SD or final question format changed")


def collect(manifest, root, references, score):
    development, holdout = manifest["development_case_ids"], manifest["holdout_case_ids"]
    validate_partition(development, holdout)
    if (manifest["mode"] != "holdout" or manifest["case_count"] != 800 or
        set(manifest["case_ids"]) != set(holdout) or len(set(holdout)) != 800 or
        not manifest["models"] or any(model not in ("qwen", "gpt") for model in manifest["models"]) or
        manifest["selection_policy"] != "decision_entity_link_v1" or manifest["entity_reserve_fraction"] != .25 or
        manifest["keep_fraction"] != .1 or manifest["student_rollout_tokens"] != 32 or
        manifest["selected_transfer_weight"] != 0 or manifest["align_student_teacher_prompt"] is not True or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The aligned-question holdout protocol changed")
    models, invalid, errors = {}, [], []
    for model in manifest["models"]:
        rows = []
        for case in manifest["case_ids"]:
            folder = Path(root) / "cases" / model / case
            path = folder / "result.json"
            if not path.is_file():
                if (folder / "error.json").is_file():
                    error = json.loads((folder / "error.json").read_text())
                    errors.append(dict(model=model, case=case, type=error.get("type")))
                continue
            result = json.loads(path.read_text())
            ref = references[case]
            try:
                validate_candidate(result, case, model, ref["task"])
            except (KeyError, TypeError, ValueError) as exc:
                invalid.append(dict(model=model, case=case, reason=str(exc)))
                continue
            grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
            controls = result["standalone_cartridge"]["original_question_controls"]
            rows.append(dict(case=case, task=ref["task"], length=result["length"],
                whole=grade(result["self_judge"]["native_prediction"]),
                zero=grade(controls["zero"]["prediction"]),
                teacher=grade(result["teacher"]["prediction"]), aligned=grade(result["prediction"])))
        grouped = defaultdict(list)
        for row in rows:
            grouped["task:" + row["task"]].append(row)
            grouped["length:" + str(row["length"])].append(row)
        models[model] = dict(overall=summarize(rows),
            groups={key: summarize(value) for key, value in grouped.items()}, rows=rows)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), metric="answer_focused_lexical_v2",
        mode="holdout", expected_per_model=800, cached_scores=True, online_api_calls=0,
        evaluation_answers_used_only_posthoc=True, models=models, errors=errors, invalid=invalid)


def combine(manifest, development, holdout):
    if (holdout["metric"] != "answer_focused_lexical_v2" or
        development["metric"] != holdout["metric"] or holdout["invalid"] or holdout["errors"]):
        raise ValueError("The paired reports are not valid")
    output = {}
    for model in manifest["models"]:
        dev = development["models"][model].get("candidate_rows", development["models"][model]["rows"])
        held = holdout["models"][model]["rows"]
        if (len(dev) != 200 or len(held) != 800 or len({row["case"] for row in dev + held}) != 1000 or
            {row["case"] for row in dev} != set(manifest["development_case_ids"]) or
            {row["case"] for row in held} != set(manifest["holdout_case_ids"])):
            return None
        all_rows = dev + held
        grouped = defaultdict(list)
        for row in all_rows:
            grouped["task:" + row["task"]].append(row)
            grouped["length:" + str(row["length"])].append(row)
        output[model] = dict(overall=summarize(all_rows), holdout=summarize(held),
            development=summarize(dev), groups={key: summarize(value) for key, value in grouped.items()})
    return dict(scope="1,000 cases: 200 development plus 800 source-disjoint holdout per eligible model",
        metric=holdout["metric"], development_used_for_configuration_selection=True,
        teacher_and_cartridge_formats_differ=True, previously_viewed_holdout_sources=2,
        evaluation_answers_used_only_posthoc=True, models=output)


def render(report, combined=False):
    lines = ["# Aligned-question SD — " + ("1,000 paired cases" if combined else "source-disjoint holdout"), "",
        "| Model | n | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher | New right / wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    for model, item in report["models"].items():
        stats = item["overall"]; accuracy = stats["accuracy"]; pair = stats["aligned_vs_teacher"]
        lines.append(f"| {model} | {stats['n']} | " + " | ".join(fmt(accuracy[key])
            for key in ("whole", "zero", "teacher", "aligned")) +
            f" | {fmt(pair['delta_pp'])} | {pair['new_correct']} / {pair['new_wrong']} |")
    lines += ["", "The Teacher has an additional generic response-format instruction; the "
        "Cartridge uses the original question. All predictions share the same task-aware loose scorer. "
        "The same-format zero comparison isolates training; references are used only after generation.",
        "", ("This combined result includes 200 development and 800 held-out cases; development was used "
             "to select the configuration. Two holdout source questions had been inspected previously." if combined else
             "This is an 800-case source-disjoint holdout. Two source questions had been inspected previously."), ""]
    return "\n".join(lines)


def write(report, output, combined=False):
    for suffix, text in ((".json", json.dumps(report, indent=2) + "\n"), (".md", render(report, combined))):
        path = output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidate-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected evaluation metric")
    write(collect(json.loads(args.manifest.read_text()), args.candidate_root,
                  json.loads(args.references.read_text()), score_prediction), args.output)


if __name__ == "__main__":
    main()
