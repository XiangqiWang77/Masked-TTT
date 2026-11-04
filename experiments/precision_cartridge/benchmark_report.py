#!/usr/bin/env python3
"""Score same-state BF16 and FP32 selected-KV materializations after inference."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

from experiments.babilong_linked_selection.long_rollout_report import validate_new


def validate_result(result, case, model, task):
    validate_new(case, result, model, task)
    diag = result["standalone_cartridge"]["precision_paired_diagnostic"]
    if (result["training"]["gain_head_mode"] != "kv" or
        len(result["training"]["steps"]) != 2 + result["self_study"]["verified"] or not diag["passed"] or
        not diag["precision_kv_only_materialized"] or diag["precision_adapter_bytes"] != 0 or
        diag["precision_tokens"] != diag["precision_matched_adapted_tokens"] or
        diag["selected_source_bytes_before_decode"] != 0 or diag["training_bytes_before_decode"] != 0 or
        not diag["same_selected_kv"] or not diag["same_question_suffix"] or
        not diag["same_generation_limit"] or not diag["same_precision_backend"] or
        diag["reference_answers_used"] or diag["teacher_answer_in_student_prompt"] or
        not isinstance(result["self_judge"]["native_prediction"], str) or
        (model == "gpt" and result["request_chat_date"] != "2026-11-03")):
        raise ValueError("The same-state precision or independent cartridge contract failed")


def summarize(rows):
    keys = ("whole", "teacher", "bf16_zero", "bf16", "fp32_zero", "fp32")
    if any(type(row[key]) is not int or row[key] not in (0, 1) for row in rows for key in keys):
        raise ValueError("Expected binary post-generation scores")
    if any(not math.isfinite(row[key]) or row[key] < 0 for row in rows
           for key in ("bf16_kv_percent", "fp32_kv_percent")):
        raise ValueError("Invalid cartridge size")
    n = len(rows)
    def pair(new, old):
        correct = sum(row[new] and not row[old] for row in rows)
        wrong = sum(row[old] and not row[new] for row in rows)
        return dict(new_correct=correct, new_wrong=wrong,
                    delta_pp=100 * (correct - wrong) / n if n else None)
    return dict(n=n, source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
        accuracy={key: 100 * sum(row[key] for row in rows) / n if n else None for key in keys},
        fp32_vs_bf16=pair("fp32", "bf16"), fp32_vs_teacher=pair("fp32", "teacher"),
        fp32_training_effect=pair("fp32", "fp32_zero"),
        bf16_training_effect=pair("bf16", "bf16_zero"),
        bf16_kv_percent=statistics.median(row["bf16_kv_percent"] for row in rows) if n else None,
        fp32_kv_percent=statistics.median(row["fp32_kv_percent"] for row in rows) if n else None)


def collect(manifest, root, references, score):
    if (manifest["case_count"] != 56 or len(manifest["case_ids"]) != 56 or
        len(set(manifest["case_ids"])) != 56 or manifest["student_rollout_tokens"] != 32 or
        manifest["selected_transfer_weight"] != 0 or manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["keep_fraction"] != .1 or manifest["reference_answers_used_in_inference"] is not False or
        manifest["online_api_calls"] != 0 or manifest["precision_comparison"] != "same_trained_state_bf16_fp32"):
        raise ValueError("The frozen precision protocol changed")
    models, errors, invalid = {}, [], []
    for model in manifest["models"]:
        rows = []
        for case in manifest["case_ids"]:
            folder = Path(root) / "cases" / model / case
            path = folder / "result.json"
            if not path.exists():
                if (folder / "error.json").exists():
                    error = json.loads((folder / "error.json").read_text())
                    errors.append(dict(model=model, case=case, type=error.get("type")))
                continue
            result = json.loads(path.read_text())
            ref = references[case]
            try:
                validate_result(result, case, model, ref["task"])
            except (KeyError, TypeError, ValueError) as exc:
                invalid.append(dict(model=model, case=case, reason=str(exc)))
                continue
            grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
            standalone = result["standalone_cartridge"]
            control = standalone["original_question_controls"]
            diag = standalone["precision_paired_diagnostic"]
            full_bytes = result["memory"]["full_document_kv_bytes"]
            if not full_bytes:
                raise ValueError("Missing full-KV byte count")
            rows.append(dict(case=case, task=ref["task"], length=result["length"],
                whole=grade(result["self_judge"]["native_prediction"]),
                teacher=grade(result["teacher"]["prediction"]),
                bf16_zero=grade(control["zero"]["prediction"]), bf16=grade(result["prediction"]),
                fp32_zero=grade(diag["precision_zero"]["prediction"]), fp32=grade(diag["precision_prediction"]),
                bf16_kv_percent=100 * standalone["memory"]["standalone_cartridge_bytes"] / full_bytes,
                fp32_kv_percent=100 * diag["precision_cartridge_bytes"] / full_bytes))
        groups = defaultdict(list)
        for row in rows:
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        models[model] = dict(overall=summarize(rows),
                             groups={key: summarize(group) for key, group in groups.items()}, rows=rows)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), metric="answer_focused_lexical_v2",
                expected_per_model=56, cached_scores=True, online_api_calls=0,
                evaluation_answers_used_only_posthoc=True, models=models, errors=errors, invalid=invalid)


def render(report):
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    lines = ["# Same trained state: BF16 and FP32 materialization", "",
        "| Model / type | n | Whole | Teacher | BF16 zero | BF16 TTT | FP32 zero | FP32 TTT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, item in report["models"].items():
        groups = [(model, item["overall"])]+[(model+" "+name[5:], stats)
                  for name, stats in sorted(item["groups"].items()) if name.startswith("task:")]
        for name, stats in groups:
            acc = stats["accuracy"]
            lines.append(f"| {name} | {stats['n']} | " + " | ".join(fmt(acc[key])
                for key in ("whole", "teacher", "bf16_zero", "bf16", "fp32_zero", "fp32")) + " |")
    lines += ["", "The same frozen model, Teacher, selected evidence and trained state are used for both "
              "materializations; no extra optimization or online API call is made. FP32 selected Values use "
              "eager attention, while BF16 uses the normal backend, so the direct FP32/BF16 comparison also "
              "includes that numerical backend difference. The FP32 zero-update control uses the same FP32 backend.",
              "Both artifacts are materialized after the full cache is released. The paired diagnostic retains "
              "the BF16 artifact while evaluating FP32; report artifact sizes separately from paired-run peak memory.",
              "The 56-case development sample contains 14 source questions and is not a 1,000-case result.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidate-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected evaluation metric")
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root,
                     json.loads(args.references.read_text()), score_prediction)
    for suffix, text in ((".json", json.dumps(report, indent=2) + "\n"), (".md", render(report))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
