#!/usr/bin/env python3
"""Paired post-generation report for linked evidence and a fixed Value rate."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.report import summarize, validate_candidate, validate_pair


def collect(manifest, root, baseline_root, references, score):
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    mode = manifest["mode"]
    if (mode not in ("development", "holdout") or
        manifest["models"] != ["gpt"] or manifest["case_count"] != (200 if mode == "development" else 800) or
        manifest["case_ids"] != manifest["development_case_ids" if mode == "development" else "holdout_case_ids"] or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["entity_reserve_fraction"] != .25 or manifest["keep_fraction"] != .1 or
        manifest["fixed_value_learning_rate"] != .001 or manifest["student_rollout_tokens"] != 8 or
        manifest["gain_head_mode"] != "kv" or manifest["max_answer_steps"] != 8 or
        manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The fixed linked Value-rate protocol changed")

    rows, errors, invalid, unmatched = [], [], [], []
    for case in manifest["case_ids"]:
        folder = Path(root) / "cases/gpt" / case
        path = folder / "result.json"
        if not path.exists():
            if (folder / "error.json").exists():
                errors.append(dict(case=case, type=json.loads((folder / "error.json").read_text()).get("type")))
            continue
        result, ref = json.loads(path.read_text()), references[case]
        try:
            validate_candidate(result, case, "gpt", ref["task"], gain_head_mode="kv")
            training = result["training"]
            if (training["value_learning_rate"] != .001 or training["gain_learning_rate"] != .01 or
                training["adaptive_max_answer_steps"] != 8 or training["adaptive_answer_KL_stop"] != .5 or
                len(training["steps"]) != 2 + result["self_study"]["verified"] or
                training.get("student_question_matches_teacher") or
                result["standalone_cartridge"].get("answer_format") != "original"):
                raise ValueError("The original-SD rate or output format changed")
        except (ValueError, KeyError, TypeError) as exc:
            invalid.append(dict(case=case, reason=str(exc)))
            continue
        grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
        controls = result["standalone_cartridge"]["original_question_controls"]
        full_bytes = result["memory"]["full_document_kv_bytes"]
        card_bytes = result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]
        if not full_bytes or not card_bytes:
            raise ValueError("Missing cache size")
        row = dict(case=case, task=ref["task"], length=result["length"],
                   whole=grade(result["self_judge"]["native_prediction"]),
                   zero=grade(controls["zero"]["prediction"]), teacher=grade(result["teacher"]["prediction"]),
                   cartridge=grade(result["prediction"]), baseline=None,
                   cartridge_percent=100 * card_bytes / full_bytes)
        baseline_path = Path(baseline_root) / "cases/gpt" / case / "result.json"
        if baseline_path.is_file():
            try:
                old = json.loads(baseline_path.read_text())
                validate_pair(old, result, case)
                row["baseline"] = grade(old["standalone_cartridge"]["paired_prompt_control"]
                                       ["trained_original"]["prediction"])
            except (ValueError, KeyError, TypeError) as exc:
                unmatched.append(dict(case=case, reason=str(exc)))
        else:
            unmatched.append(dict(case=case, reason="baseline_missing"))
        rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=mode, model="gpt",
                expected=manifest["case_count"], fixed_value_learning_rate=.001,
                metric="answer_focused_lexical_v2", cached_scores=True, online_api_calls=0,
                reference_answers_used_only_posthoc=True, overall=summarize(rows),
                groups={name: summarize(items) for name, items in sorted(groups.items())},
                rows=rows, errors=errors, invalid=invalid, unmatched_baselines=unmatched)


def render(report):
    item = report["overall"]
    acc = item["accuracy"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    pair = item["strictly_paired"]
    return "\n".join([
        "# GPT-OSS: linked evidence, original SD and fixed Value rate", "",
        "| Cases | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {item['n']} | {fmt(acc['whole'])} | {fmt(acc['zero'])} | {fmt(acc['teacher'])} | "
        f"{fmt(acc['cartridge'])} | {fmt(item['versus_teacher']['delta_pp'])} pp |", "",
        f"Matched original Value-rate baseline: {pair['n']} cases; "
        f"{pair['new_correct']} new correct / {pair['new_wrong']} new wrong; "
        f"{fmt(pair['delta_pp'])} pp.", "",
        "Teacher has the original generic answer instruction; zero-update and cartridge use the "
        "same original-question format. Scoring is a shared loose lexical metric after generation.",
        "The final cartridge decodes only after full cache, selected source cache, and training state are released.", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidate-root", "baseline-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected answer metric")
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root, args.baseline_root,
                     json.loads(args.references.read_text()), score_prediction)
    for suffix, value in ((".json", json.dumps(report, indent=2) + "\n"), (".md", render(report))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)


if __name__ == "__main__":
    main()
