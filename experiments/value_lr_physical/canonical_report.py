#!/usr/bin/env python3
"""Score the same-device canonical-evidence comparison after generation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.report import validate_candidate

ARMS = ("whole", "original_zero", "original_teacher", "original_cartridge",
        "canonical_zero", "canonical_teacher", "canonical_cartridge")


def summarize(rows):
    n = len(rows)
    if any(type(row.get(name)) is not int or row[name] not in (0, 1)
           for row in rows for name in ARMS):
        raise ValueError("Every paired answer must have a binary post-generation score")
    if any(type(row.get(name)) is not bool for row in rows for name in
           ("same_teacher_tokens", "same_zero_tokens", "same_practice_checks", "teacher_prompt_length_changed")):
        raise ValueError("Every paired tokenization indicator must be boolean")
    comparisons = {}
    for base in ("whole", "original_cartridge", "canonical_zero", "canonical_teacher"):
        good = sum(not row[base] and row["canonical_cartridge"] for row in rows)
        bad = sum(row[base] and not row["canonical_cartridge"] for row in rows)
        comparisons[base] = dict(new_correct=good, new_wrong=bad,
                                 delta_pp=100 * (good-bad) / n if n else None)
    return dict(n=n, accuracy={name: 100 * sum(row[name] for row in rows) / n if n else None
                              for name in ARMS}, versus=comparisons,
                teacher_tokens_changed=sum(not row["same_teacher_tokens"] for row in rows),
                zero_tokens_changed=sum(not row["same_zero_tokens"] for row in rows),
                practice_checks_changed=sum(not row["same_practice_checks"] for row in rows),
                teacher_prompt_length_changed=sum(row["teacher_prompt_length_changed"] for row in rows))


def development_decision(report, model):
    item = report["models"][model]
    stats = item["overall"]
    if (report["mode"] != "development" or report["expected_per_model"] != 200 or stats["n"] != 200 or
        report["metric"] != "answer_focused_lexical_v2" or
        any(error["model"] == model for error in report["errors"] + report["invalid"])):
        return dict(eligible=False, reason="development_incomplete_or_invalid")
    delta = {name: comparison["delta_pp"] for name, comparison in stats["versus"].items()}
    eligible = all(value > 0 for value in delta.values())
    return dict(eligible=eligible, reason="beats_same_run_controls" if eligible else
                "does_not_beat_same_run_controls", delta_pp=delta)


def collect(manifest, root, references, score):
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    mode = manifest["mode"]
    if (mode not in ("development", "holdout") or not manifest["models"] or
        any(model not in ("qwen", "gpt") for model in manifest["models"]) or
        manifest["case_count"] != (200 if mode == "development" else 800) or
        manifest["case_ids"] != manifest["development_case_ids" if mode == "development" else "holdout_case_ids"] or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["entity_reserve_fraction"] != .25 or manifest["keep_fraction"] != .1 or
        manifest["evidence_tokenization_comparison"] != "original_vs_canonical_fragment" or
        manifest["student_rollout_tokens"] != 8 or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The frozen canonical-evidence experiment changed")
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
                validate_candidate(result, case, model, ref["task"], gain_head_mode="kv")
                p, t = result["canonical_tokenization_pair"], result["training"]
                if (t["gain_learning_rate"] != .01 or t["value_learning_rate"] != .0001 or
                    t["adaptive_max_answer_steps"] != 8 or t["student_question_matches_teacher"] or
                    len(t["steps"]) != 2 + result["self_study"]["verified"] or
                    p["candidate_verified"] != result["self_study"]["verified"] or
                    p["baseline_native_prediction"] != result["self_judge"]["native_prediction"] or
                    p["reference_answers_used"] or p["teacher_probability_tensors_compared"] or
                    not all(p[key] for key in ("same_loaded_model", "same_gpu", "same_request",
                        "independently_prefilled_caches", "same_selection", "baseline_standalone_released",
                        "candidate_standalone_released")) or
                    any(type(p[key]) is not bool for key in
                        ("same_practice_checks", "same_teacher_tokens", "same_zero_tokens"))):
                    raise ValueError("The selected-evidence pair does not satisfy the frozen protocol")
            except (KeyError, TypeError, ValueError) as exc:
                invalid.append(dict(model=model, case=case, reason=str(exc)))
                continue
            def grade(text):
                value = score(text, ref["answers"], "babilong", ref["task"])[0]
                if type(value) not in (int, float) or value not in (0, 1):
                    raise ValueError("The shared scorer must return a binary BABILONG score")
                return int(value)
            controls = result["standalone_cartridge"]["original_question_controls"]
            rows.append(dict(case=case, task=ref["task"], length=result["length"],
                whole=grade(result["self_judge"]["native_prediction"]),
                original_zero=grade(p["baseline_zero_prediction"]),
                original_teacher=grade(p["baseline_teacher_prediction"]),
                original_cartridge=grade(p["baseline_prediction"]),
                canonical_zero=grade(controls["zero"]["prediction"]),
                canonical_teacher=grade(result["teacher"]["prediction"]),
                canonical_cartridge=grade(result["prediction"]),
                same_teacher_tokens=p["same_teacher_tokens"], same_zero_tokens=p["same_zero_tokens"],
                same_practice_checks=p["same_practice_checks"],
                teacher_prompt_length_changed=p["baseline_teacher_prompt_tokens"] != p["candidate_teacher_prompt_tokens"]))
        groups = defaultdict(list)
        for row in rows:
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        models[model] = dict(overall=summarize(rows),
                             groups={name: summarize(items) for name, items in sorted(groups.items())}, rows=rows)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=mode,
        expected_per_model=manifest["case_count"], metric="answer_focused_lexical_v2", cached_scores=True,
        online_api_calls=0, reference_answers_used_only_posthoc=True, models=models, errors=errors, invalid=invalid)


def render(report):
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    lines = ["# Same-device selected-evidence tokenization comparison", "",
        "| Model | n | Whole | Original zero | Original Teacher | Original SD | Canonical zero | Canonical Teacher | Canonical SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, item in report["models"].items():
        stats, a = item["overall"], item["overall"]["accuracy"]
        lines.append(f"| {model} | {stats['n']} | " + " | ".join(fmt(a[key]) for key in ARMS) + " |")
    lines += ["", "| Model | vs original SD: new right / wrong | vs own Teacher: new right / wrong |",
              "|---|---:|---:|"]
    for model, item in report["models"].items():
        v = item["overall"]["versus"]
        lines.append(f"| {model} | {v['original_cartridge']['new_correct']} / {v['original_cartridge']['new_wrong']} | "
                     f"{v['canonical_teacher']['new_correct']} / {v['canonical_teacher']['new_wrong']} |")
    lines += ["", "Both arms use the same selected original chunks and original on-policy SD. "
        "Teacher and cartridge are independently prefetched in each arm; changing evidence tokens can change "
        "the Teacher and zero-update answer. References are used only for post-generation loose lexical scoring.", ""]
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
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root,
                     json.loads(args.references.read_text()), score_prediction)
    for suffix, text in ((".json", json.dumps(report, indent=2) + "\n"), (".md", render(report))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
