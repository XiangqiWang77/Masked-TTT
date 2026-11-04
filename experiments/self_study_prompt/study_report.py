#!/usr/bin/env python3
"""Score completed same-GPU study-prompt pairs after generation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.report import summarize, validate_candidate


def collect(manifest, root, references, score):
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    mode = manifest["mode"]
    if (mode not in ("development", "holdout") or manifest["models"] != ["qwen"] or
        manifest["case_count"] != (200 if mode == "development" else 800) or
        manifest["case_ids"] != manifest["development_case_ids" if mode == "development" else "holdout_case_ids"] or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["entity_reserve_fraction"] != .25 or manifest["keep_fraction"] != .1 or
        manifest["self_study_prompt_comparison"] != "original_vs_study_only" or
        manifest["student_rollout_tokens"] != 8 or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The frozen study-prompt experiment changed")
    rows, errors, invalid = [], [], []
    for case in manifest["case_ids"]:
        folder = Path(root) / "cases/qwen" / case
        path = folder / "result.json"
        if not path.is_file():
            if (folder / "error.json").is_file():
                errors.append(dict(case=case, type=json.loads((folder / "error.json").read_text()).get("type")))
            continue
        result, ref = json.loads(path.read_text()), references[case]
        try:
            validate_candidate(result, case, "qwen", ref["task"], gain_head_mode="kv")
            p, t = result["study_prompt_pair"], result["training"]
            if (t["gain_learning_rate"] != .01 or t["value_learning_rate"] != .0001 or
                t["adaptive_max_answer_steps"] != 8 or t["student_question_matches_teacher"] or
                len(t["steps"]) != 2 + result["self_study"]["verified"] or
                p["baseline_training_steps"] != 2 + p["baseline_verified"] or
                p["candidate_training_steps"] != 2 + p["candidate_verified"] or
                p["candidate_verified"] != result["self_study"]["verified"] or
                p["reference_answers_used"] or p["teacher_probability_tensors_compared"] or
                p["baseline_native_prediction"] != result["self_judge"]["native_prediction"] or
                not all(p[key] for key in ("same_loaded_model", "same_gpu", "same_request",
                    "independently_prefilled_caches", "same_selection", "same_teacher_tokens",
                    "same_zero_tokens", "baseline_standalone_released", "candidate_standalone_released"))):
                raise ValueError("The matched original-method arms changed")
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(dict(case=case, reason=str(exc)))
            continue
        grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
        controls = result["standalone_cartridge"]["original_question_controls"]
        full_bytes, card_bytes = (result["memory"]["full_document_kv_bytes"],
                                  result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"])
        if not full_bytes or not card_bytes:
            raise ValueError("Missing full or cartridge KV size")
        rows.append(dict(case=case, task=ref["task"], length=result["length"],
            whole=grade(result["self_judge"]["native_prediction"]),
            zero=grade(controls["zero"]["prediction"]), teacher=grade(result["teacher"]["prediction"]),
            baseline=grade(p["baseline_prediction"]), cartridge=grade(result["prediction"]),
            baseline_verified=p["baseline_verified"], candidate_verified=p["candidate_verified"],
            same_practice_checks=p["same_practice_checks"], cartridge_percent=100 * card_bytes / full_bytes))
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    stats = summarize(rows)
    practice = dict(original=sum(row["baseline_verified"] for row in rows),
                    study_only=sum(row["candidate_verified"] for row in rows),
                    more=sum(row["candidate_verified"] > row["baseline_verified"] for row in rows),
                    fewer=sum(row["candidate_verified"] < row["baseline_verified"] for row in rows),
                    identical_checks=sum(row["same_practice_checks"] for row in rows))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=mode, model="qwen",
        expected=manifest["case_count"], metric="answer_focused_lexical_v2", cached_scores=True,
        online_api_calls=0, reference_answers_used_only_posthoc=True, overall=stats, practice=practice,
        groups={name: summarize(items) for name, items in sorted(groups.items())},
        rows=rows, errors=errors, invalid=invalid)


def render(report):
    stats, practice = report["overall"], report["practice"]
    accuracy, pair = stats["accuracy"], stats["strictly_paired"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    return "\n".join([
        "# Qwen: original versus study-only exercise prompts", "",
        "| Cases | Whole | Selected zero | ICL Teacher | Original SD | Study-only SD | vs Teacher | vs original SD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {stats['n']} | {fmt(accuracy['whole'])} | {fmt(accuracy['zero'])} | {fmt(accuracy['teacher'])} | "
        f"{fmt(stats['paired_baseline_accuracy'])} | {fmt(accuracy['cartridge'])} | "
        f"{fmt(stats['versus_teacher']['delta_pp'])} pp | {fmt(pair['delta_pp'])} pp |", "",
        f"Versus original SD: {pair['new_correct']} new correct / {pair['new_wrong']} new wrong.",
        f"Verified exercises: {practice['original']} original, {practice['study_only']} study-only; "
        f"more / fewer on {practice['more']} / {practice['fewer']} cases.", "",
        "The same question and selected evidence are run on the same loaded model and GPU. "
        "Each arm independently prefills, trains, materializes and releases its own source caches.",
        "Teacher and zero-update output tokens must match. Their full probability tensors are not compared.",
        "ICL Teacher uses the generic response instruction; both cartridges use the original question format. "
        "References are used only after generation with the shared loose lexical scorer.", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidate-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected answer metric")
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root,
                     json.loads(args.references.read_text()), score_prediction)
    for suffix, text in ((".json", json.dumps(report, indent=2) + "\n"), (".md", render(report))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
