#!/usr/bin/env python3
"""Score a stronger GPT-OSS Value rate against its fixed paired baseline."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import summarize, validate_partition
from experiments.value_lr_physical.report import validate_result


def same_baseline(candidate, baseline):
    control = candidate["standalone_cartridge"]["original_question_controls"]
    old = baseline["standalone_cartridge"]["paired_prompt_control"]
    if (candidate["key"] != baseline["key"] or candidate["model"] != baseline["model"] or
        candidate["selection"]["selected_original_chunks"] != baseline["selection"]["selected_original_chunks"] or
        candidate["teacher"]["tokens"] != baseline["teacher"]["tokens"] or
        candidate["self_study"]["checks"] != baseline["self_study"]["checks"] or
        candidate["training"]["steps"][0]["student_rollout_tokens"] != baseline["training"]["steps"][0]["student_rollout_tokens"] or
        control["zero"]["tokens"] != old["zero_original"]["tokens"] or
        baseline["value_learning_rate"] != .001 or baseline["gain_learning_rate"] != .01 or
        baseline["training"]["objective"] != "on-policy KL(teacher || student)" or
        baseline["training"]["selected_transfer_weight"] != 0):
        raise ValueError("Baseline selection, Teacher, practice, initial prefix or zero control differs")


def paired(rows):
    pairs = [(row["baseline"], row["cartridge"]) for row in rows if row["baseline"] is not None]
    if any(type(a) is not int or type(b) is not int or a not in (0, 1) or b not in (0, 1) for a, b in pairs):
        raise ValueError("Expected binary answer scores")
    n = len(pairs)
    gain = sum(not a and b for a, b in pairs)
    loss = sum(a and not b for a, b in pairs)
    return dict(n=n, baseline_accuracy=100 * sum(a for a, _ in pairs) / n if n else None,
                candidate_accuracy=100 * sum(b for _, b in pairs) / n if n else None,
                new_correct=gain, new_wrong=loss, delta_pp=100 * (gain-loss) / n if n else None)


def collect(manifest, root, baseline_root, references, score):
    development, holdout = manifest["development_case_ids"], manifest["holdout_case_ids"]
    validate_partition(development, holdout)
    mode = manifest["mode"]
    if (mode not in ("development", "holdout") or
        manifest["case_count"] != (200 if mode == "development" else 800) or
        set(manifest["case_ids"]) != set(development if mode == "development" else holdout) or
        manifest["models"] != ["gpt"] or manifest["selection_policy"] != "cached_decision_score10" or
        manifest["fixed_value_learning_rate"] != .003 or manifest["student_rollout_tokens"] != 8 or
        manifest["selected_transfer_weight"] != 0 or manifest["reference_answers_used_in_inference"] is not False or
        manifest["online_api_calls"] != 0 or manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The fixed stronger-Value protocol changed")
    rows, errors, invalid, unmatched = [], [], [], []
    for case in manifest["case_ids"]:
        folder = Path(root) / "cases/gpt" / case
        path = folder / "result.json"
        if not path.exists():
            if (folder / "error.json").is_file():
                error = json.loads((folder / "error.json").read_text())
                errors.append(dict(case=case, type=error.get("type")))
            continue
        result, ref = json.loads(path.read_text()), references[case]
        try:
            validate_result(result, case, "gpt", ref["task"], .003)
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(dict(case=case, reason=str(exc)))
            continue
        grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
        control = result["standalone_cartridge"]["original_question_controls"]
        full_bytes = result["memory"]["full_document_kv_bytes"]
        card_bytes = result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]
        if not full_bytes or not card_bytes:
            raise ValueError("Missing document cache accounting")
        row = dict(case=case, task=ref["task"], length=result["length"],
            whole=grade(result["self_judge"]["native_prediction"]), zero=grade(control["zero"]["prediction"]),
            teacher=grade(result["teacher"]["prediction"]), cartridge=grade(result["prediction"]),
            linked8=None, baseline=None, cartridge_percent=100 * card_bytes / full_bytes)
        if mode == "development":
            baseline_path = Path(baseline_root) / "cases/gpt" / case / "lr_0.0010.json"
            if baseline_path.is_file():
                old = json.loads(baseline_path.read_text())
                try:
                    same_baseline(result, old)
                    row["baseline"] = grade(old["standalone_cartridge"]["paired_prompt_control"]
                                           ["trained_original"]["prediction"])
                except (KeyError, TypeError, ValueError) as exc:
                    unmatched.append(dict(case=case, reason=str(exc)))
            else:
                unmatched.append(dict(case=case, reason="baseline_missing"))
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=mode, model="gpt",
        expected=manifest["case_count"], fixed_value_learning_rate=.003,
        metric="answer_focused_lexical_v2", cached_scores=True, online_api_calls=0,
        reference_answers_used_only_posthoc=True, overall=summarize(rows),
        versus_fixed_baseline=paired(rows), groups={name: summarize(items) for name, items in groups.items()},
        rows=rows, errors=errors, invalid=invalid, unmatched_baselines=unmatched)


def render(data):
    item, pair = data["overall"], data["versus_fixed_baseline"]
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    a = item["accuracy"]
    lines = ["# GPT-OSS: original SD with stronger Value rate", "",
        "| Model | n | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| GPT-OSS-20B | {item['n']} | {fmt(a['whole'])} | {fmt(a['zero'])} | {fmt(a['teacher'])} | "
        f"{fmt(a['cartridge'])} | {fmt(item['versus_teacher']['delta_pp'])} |", ""]
    if data["mode"] == "development":
        lines += [f"Same-selection and same-start matched comparison against 0.001: {pair['n']} cases, "
                  f"{pair['new_correct']} new correct / {pair['new_wrong']} new wrong; "
                  f"{fmt(pair['delta_pp'])} percentage points.", ""]
    lines += ["Value rate 0.003 is fixed before this run. The Teacher has the original generic response "
              "instruction; the selected zero and cartridge use the same original question. "
              "All reference answers are used only for post-generation scoring.", ""]
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
    result = collect(json.loads(args.manifest.read_text()), args.candidate_root, args.baseline_root,
                     json.loads(args.references.read_text()), score_prediction)
    for suffix, text in ((".json", json.dumps(result, indent=2) + "\n"), (".md", render(result))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
