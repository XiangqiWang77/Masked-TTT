#!/usr/bin/env python3
"""Score same-GPU gain-rate pairs only after their answers are generated."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.report import summarize, validate_candidate
from experiments.gain_rate_pair.protocol import BASE_GAIN, CANDIDATE_GAIN, VALUE_RATES


def check_manifest(manifest):
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    mode = manifest["mode"]
    models = manifest["models"]
    if (mode not in ("development", "holdout") or not models or len(models) != len(set(models)) or
        any(model not in VALUE_RATES for model in models) or
        manifest["case_count"] != (200 if mode == "development" else 800) or
        manifest["case_ids"] != manifest["development_case_ids" if mode == "development" else "holdout_case_ids"] or
        manifest["selection_policy"] != "decision_entity_link_v1" or manifest["entity_reserve_fraction"] != .25 or
        manifest["keep_fraction"] != .1 or manifest["student_rollout_tokens"] != 8 or
        manifest["baseline_gain_lr"] != BASE_GAIN or manifest["candidate_gain_lr"] != CANDIDATE_GAIN or
        manifest["value_learning_rates"] != VALUE_RATES or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The paired gain-rate experiment protocol changed")


def validate_pair(result, case, model, task):
    validate_candidate(result, case, model, task, gain_head_mode="kv")
    p, t = result["gain_rate_pair"], result["training"]
    if (p["model"] != model or t["gain_learning_rate"] != CANDIDATE_GAIN or
        t["value_learning_rate"] != VALUE_RATES[model] or p["value_lr"] != VALUE_RATES[model] or
        p["baseline_gain_lr"] != BASE_GAIN or p["candidate_gain_lr"] != CANDIDATE_GAIN or
        t["adaptive_max_answer_steps"] != 8 or t["student_question_matches_teacher"] or
        len(t["steps"]) != 2 + result["self_study"]["verified"] or
        p["baseline_training_steps"] != 2 + p["baseline_verified"] or
        p["candidate_training_steps"] != 2 + p["candidate_verified"] or
        p["candidate_verified"] != result["self_study"]["verified"] or
        p["baseline_native_prediction"] != result["self_judge"]["native_prediction"] or
        p["changed_settings"] != ["gain_learning_rate"] or
        p["reference_answers_used"] or p["teacher_probability_tensors_compared"] or
        not all(p[key] for key in ("same_loaded_model", "same_gpu", "same_request", "same_practice_checks",
            "independently_prefilled_caches", "same_selection", "same_teacher_tokens", "same_zero_tokens",
            "baseline_standalone_released", "candidate_standalone_released"))):
        raise ValueError("The same-GPU gain-rate pair changed")


def collect(manifest, root, references, score):
    check_manifest(manifest)
    models, errors, invalid = {}, [], []
    for model in manifest["models"]:
        rows = []
        for case in manifest["case_ids"]:
            folder = Path(root) / "cases" / model / case
            path = folder / "result.json"
            if not path.is_file():
                if (folder / "error.json").is_file():
                    errors.append(dict(model=model, case=case,
                                       type=json.loads((folder / "error.json").read_text()).get("type")))
                continue
            result, ref = json.loads(path.read_text()), references[case]
            try:
                validate_pair(result, case, model, ref["task"])
            except (KeyError, TypeError, ValueError) as exc:
                invalid.append(dict(model=model, case=case, reason=str(exc)))
                continue
            grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
            p, controls = result["gain_rate_pair"], result["standalone_cartridge"]["original_question_controls"]
            full = result["memory"]["full_document_kv_bytes"]
            card = result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]
            if not full or not card:
                raise ValueError("Missing full or cartridge KV size")
            rows.append(dict(case=case, task=ref["task"], length=result["length"],
                whole=grade(result["self_judge"]["native_prediction"]), zero=grade(controls["zero"]["prediction"]),
                teacher=grade(result["teacher"]["prediction"]), baseline=grade(p["baseline_prediction"]),
                cartridge=grade(result["prediction"]), cartridge_percent=100*card/full))
        groups = defaultdict(list)
        for row in rows:
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        models[model] = dict(overall=summarize(rows),
            groups={name: summarize(items) for name, items in sorted(groups.items())}, rows=rows)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=manifest["mode"],
        expected_per_model=manifest["case_count"], metric="answer_focused_lexical_v2", cached_scores=True,
        online_api_calls=0, reference_answers_used_only_posthoc=True, models=models, errors=errors, invalid=invalid)


def render(report):
    fmt = lambda n: "—" if n is None else f"{n:.2f}"
    lines = ["# Original SD: same-GPU attention-gain learning rates", "",
        "| Model | Cases | Whole | Selected zero | ICL Teacher | Gain 0.01 | Gain 0.03 | Δ vs Teacher | Δ vs 0.01 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    notes = []
    for model, entry in report["models"].items():
        stats = entry["overall"]
        a, p = stats["accuracy"], stats["strictly_paired"]
        lines.append(f"| {model} | {stats['n']} | {fmt(a['whole'])} | {fmt(a['zero'])} | "
            f"{fmt(a['teacher'])} | {fmt(stats['paired_baseline_accuracy'])} | {fmt(a['cartridge'])} | "
            f"{fmt(stats['versus_teacher']['delta_pp'])} | {fmt(p['delta_pp'])} |")
        notes.append(f"{model}: vs 0.01, {p['new_correct']} new correct / {p['new_wrong']} new wrong.")
    lines += ["", *notes, "", "Within each model, both arms use the same GPU, selected evidence, Teacher output tokens, "
        "verified practices and zero-update answer. Full Teacher probability tensors are not compared.",
        "Full Student SD and independent selected-KV materialization are unchanged; only the gain learning rate "
        "differs within a pair. Teacher and cartridge have different response-format instructions.",
        "All arms use the same loose lexical score; evaluation references are used only after generation.", ""]
    return "\n".join(lines)


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
