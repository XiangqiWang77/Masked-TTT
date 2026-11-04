#!/usr/bin/env python3
"""Score and aggregate completed Qwen same-GPU SD hyperparameter pairs."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.babilong_linked_selection.long_rollout_report import validate_new
from experiments.gqa_head_gain.report import summarize


def check_manifest(manifest):
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    mode = manifest["mode"]
    if (mode not in ("development", "holdout") or manifest["models"] != ["qwen"] or
        manifest["case_count"] != (200 if mode == "development" else 800) or
        manifest["case_ids"] != manifest["development_case_ids" if mode == "development" else "holdout_case_ids"] or
        manifest["selection_policy"] != "decision_entity_link_v1" or manifest["entity_reserve_fraction"] != .25 or
        manifest["keep_fraction"] != .1 or manifest["student_rollout_tokens"] != 32 or
        manifest["baseline_rollout_tokens"] != 8 or manifest["candidate_value_lr"] != .001 or
        manifest["baseline_value_lr"] != .0001 or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The paired original-SD experiment protocol changed")


def validate_pair(result, case, task):
    validate_new(case, result, "qwen", task)
    p, t = result["rollout_value_pair"], result["training"]
    if (t["gain_head_mode"] != "kv" or t["gain_learning_rate"] != .01 or t["value_learning_rate"] != .001 or
        t["adaptive_max_answer_steps"] != 8 or t["student_question_matches_teacher"] or
        len(t["steps"]) != 2 + result["self_study"]["verified"] or
        p["baseline_training_steps"] != 2 + p["baseline_verified"] or
        p["candidate_training_steps"] != 2 + p["candidate_verified"] or
        p["baseline_rollout_tokens"] != 8 or p["baseline_value_lr"] != .0001 or
        p["candidate_verified"] != result["self_study"]["verified"] or
        p["baseline_native_prediction"] != result["self_judge"]["native_prediction"] or
        set(p["changed_settings"]) != {"student_rollout_tokens", "value_learning_rate"} or
        p["reference_answers_used"] or p["teacher_probability_tensors_compared"] or
        not all(p[key] for key in ("same_loaded_model", "same_gpu", "same_request", "same_practice_checks",
            "independently_prefilled_caches", "same_selection", "same_teacher_tokens", "same_zero_tokens",
            "baseline_standalone_released", "candidate_standalone_released"))):
        raise ValueError("The same-GPU original-SD hyperparameter pair changed")


def collect(manifest, root, references, score):
    check_manifest(manifest)
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
            validate_pair(result, case, ref["task"])
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(dict(case=case, reason=str(exc)))
            continue
        grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
        p, controls = result["rollout_value_pair"], result["standalone_cartridge"]["original_question_controls"]
        full_bytes = result["memory"]["full_document_kv_bytes"]
        card_bytes = result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]
        if not full_bytes or not card_bytes:
            raise ValueError("Missing full or cartridge KV size")
        rows.append(dict(case=case, task=ref["task"], length=result["length"],
            whole=grade(result["self_judge"]["native_prediction"]), zero=grade(controls["zero"]["prediction"]),
            teacher=grade(result["teacher"]["prediction"]), baseline=grade(p["baseline_prediction"]),
            cartridge=grade(result["prediction"]), cartridge_percent=100*card_bytes/full_bytes))
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=manifest["mode"], model="qwen",
        expected=manifest["case_count"], metric="answer_focused_lexical_v2", cached_scores=True,
        online_api_calls=0, reference_answers_used_only_posthoc=True, overall=summarize(rows),
        groups={name: summarize(items) for name, items in sorted(groups.items())},
        rows=rows, errors=errors, invalid=invalid)


def render(report):
    s = report["overall"]
    a, paired = s["accuracy"], s["strictly_paired"]
    fmt = lambda n: "—" if n is None else f"{n:.2f}"
    return "\n".join(["# Qwen: original SD, paired rollout and Value learning rate", "",
        "| Cases | Whole | Selected zero | ICL Teacher | 8 / 0.0001 | 32 / 0.001 | Δ vs Teacher | Δ vs 8 / 0.0001 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {s['n']} | {fmt(a['whole'])} | {fmt(a['zero'])} | {fmt(a['teacher'])} | "
        f"{fmt(s['paired_baseline_accuracy'])} | {fmt(a['cartridge'])} | "
        f"{fmt(s['versus_teacher']['delta_pp'])} pp | {fmt(paired['delta_pp'])} pp |", "",
        f"Versus baseline: {paired['new_correct']} new correct and {paired['new_wrong']} new wrong.",
        "Both arms share the same loaded model, GPU, selected evidence, Teacher tokens, verified practices "
        "and zero-update answer. Their full Teacher probability tensors are not compared.",
        "Each arm separately prefills the two views, applies on-policy Teacher→Student KL, materializes its "
        "selected KV and releases source caches before the final answer.",
        "Teacher uses the generic response instruction; both cartridges use the original question format. "
        "References are used only after generation under the common loose lexical scorer.", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidate-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected answer metric")
    data = collect(json.loads(args.manifest.read_text()), args.candidate_root,
                   json.loads(args.references.read_text()), score_prediction)
    for suffix, contents in ((".json", json.dumps(data, indent=2) + "\n"), (".md", render(data))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(contents)
        temporary.replace(path)


if __name__ == "__main__":
    main()
