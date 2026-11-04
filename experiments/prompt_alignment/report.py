#!/usr/bin/env python3
"""Grade aligned-question SD against matched two-step checkpoint answers."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_report import validate_new
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.long_rollout_steps.report import validate_result as validate_steps


def validate_pair(candidate, baseline, case, model, task):
    validate_new(case, candidate, model, task)
    validate_steps(baseline, case, model, task)
    train = candidate["training"]
    if (not train.get("student_question_matches_teacher") or
        train["gain_head_mode"] != "kv" or
        len(train["steps"]) != 2 + candidate["self_study"]["verified"]):
        raise ValueError("Unexpected aligned-question candidate")
    fields = ("selected_original_chunks", "selected_chunks", "total_chunks", "selected_tokens")
    if any(candidate["selection"][key] != baseline["selection"][key] for key in fields):
        raise ValueError("Selected chunks differ")
    if candidate["teacher"]["tokens"] != baseline["teacher"]["tokens"]:
        raise ValueError("Teacher output differs")
    if candidate["self_study"]["checks"] != baseline["self_study"]["checks"]:
        raise ValueError("Practice inputs or verification differ")
    can = candidate["standalone_cartridge"]["original_question_controls"]
    old = baseline["standalone_cartridge"]["original_question_controls"]
    if can["zero"]["tokens"] != old["zero"]["tokens"]:
        raise ValueError("Same-format zero-update answer differs")
    if candidate["standalone_cartridge"].get("answer_format") != "original":
        raise ValueError("Final question format changed")


def summarize(rows):
    n = len(rows)
    fields = ("whole", "zero", "teacher", "baseline", "aligned")
    if any(type(row[key]) is not int or row[key] not in (0, 1) for row in rows for key in fields):
        raise ValueError("Scores must be binary")
    def paired(new, old):
        correct = sum(row[new] and not row[old] for row in rows)
        wrong = sum(row[old] and not row[new] for row in rows)
        return dict(new_correct=correct, new_wrong=wrong,
                    delta_pp=100 * (correct - wrong) / n if n else None)
    return dict(n=n, source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
                accuracy={key: 100 * sum(row[key] for row in rows) / n if n else None
                          for key in fields},
                aligned_vs_baseline=paired("aligned", "baseline"),
                aligned_vs_teacher=paired("aligned", "teacher"),
                aligned_vs_zero=paired("aligned", "zero"))


def collect(manifest, root, baseline_root, references, score):
    development, holdout = manifest["development_case_ids"], manifest["holdout_case_ids"]
    validate_partition(development, holdout)
    if (manifest["mode"] != "development" or manifest["case_count"] != 200 or
        set(manifest["case_ids"]) != set(development) or len(set(development)) != 200 or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["entity_reserve_fraction"] != .25 or manifest["keep_fraction"] != .1 or
        manifest["student_rollout_tokens"] != 32 or manifest["selected_transfer_weight"] != 0 or
        manifest.get("align_student_teacher_prompt") is not True or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The aligned-question development protocol changed")
    models, invalid, errors = {}, [], []
    for model in manifest["models"]:
        rows, unmatched, completed = [], 0, 0
        for case in manifest["case_ids"]:
            folder = Path(root) / "cases" / model / case
            path = folder / "result.json"
            if not path.is_file():
                if (folder / "error.json").is_file():
                    error = json.loads((folder / "error.json").read_text())
                    errors.append(dict(model=model, case=case, type=error.get("type")))
                continue
            completed += 1
            old_path = Path(baseline_root) / "cases" / model / case / "result.json"
            if not old_path.is_file():
                unmatched += 1
                continue
            result, old = json.loads(path.read_text()), json.loads(old_path.read_text())
            ref = references[case]
            try:
                validate_pair(result, old, case, model, ref["task"])
            except (KeyError, TypeError, ValueError) as exc:
                invalid.append(dict(model=model, case=case, reason=str(exc)))
                continue
            grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
            control = result["standalone_cartridge"]["original_question_controls"]
            checkpoint = old["standalone_cartridge"]["step_probe"]["checkpoints"]["2"]
            rows.append(dict(case=case, task=ref["task"], length=result["length"],
                whole=grade(result["self_judge"]["native_prediction"]),
                zero=grade(control["zero"]["prediction"]), teacher=grade(result["teacher"]["prediction"]),
                baseline=grade(checkpoint["prediction"]), aligned=grade(result["prediction"])))
        groups = defaultdict(list)
        for row in rows:
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        models[model] = dict(candidate_completed=completed, baseline_not_yet_available=unmatched,
            overall=summarize(rows), groups={key: summarize(group) for key, group in groups.items()}, rows=rows)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), metric="answer_focused_lexical_v2",
                expected_per_model=200, cached_scores=True, online_api_calls=0,
                evaluation_answers_used_only_posthoc=True, same_main_teacher_prompt=True,
                models=models, errors=errors, invalid=invalid)


def render(report):
    lines = ["# Aligned-question on-policy SD — matched development cases", "",
             "| Model | matched n | whole | selected zero | ICL Teacher | original SD | aligned SD | new right / wrong | Δ pp |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    for model, item in report["models"].items():
        stats = item["overall"]; a = stats["accuracy"]; pair = stats["aligned_vs_baseline"]
        lines.append(f"| {model} | {stats['n']} | " + " | ".join(fmt(a[key])
            for key in ("whole", "zero", "teacher", "baseline", "aligned")) +
            f" | {pair['new_correct']} / {pair['new_wrong']} | {fmt(pair['delta_pp'])} |")
    lines += ["", "Comparison uses identical selected evidence, Teacher token output, verified practice "
              "exercises, zero-update token output, and original final question format. The baseline is "
              "the two-base-step checkpoint; the candidate is decoded after full and selected source KV "
              "and training state are released. References are used only for post-generation scoring.",
              "", "This 200-case development set contains 50 source questions with four lengths each; "
              "it is not the 1,000-case final evaluation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidate-root", "baseline-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected evaluation metric")
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root, args.baseline_root,
                     json.loads(args.references.read_text()), score_prediction)
    for suffix, text in ((".json", json.dumps(report, indent=2) + "\n"), (".md", render(report))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
