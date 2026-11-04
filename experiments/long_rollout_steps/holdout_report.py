#!/usr/bin/env python3
"""Score a fixed-step source-disjoint holdout and combine it with development."""
import argparse
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection import long_rollout_report as base
from experiments.babilong_linked_selection.long_rollout_protocol import summarize
from experiments.long_rollout_steps.holdout_execute import model_paths
from experiments.long_rollout_steps.selection import development_choices


def collect(manifest, root, references, score):
    model_paths(manifest)
    original = base.validate_new

    def validate(case, result, model, task):
        original(case, result, model, task)
        expected = manifest["fixed_final_base_steps"][model]
        training = result["training"]
        practices = result["self_study"]["verified"]
        if (training["gain_head_mode"] != "kv" or len(training["steps"]) != expected + practices or
            not isinstance(result["self_judge"]["native_prediction"], str) or
            (model == "gpt" and result["request_chat_date"] != "2026-11-03")):
            raise ValueError("A holdout request did not use its fixed development step")

    base.validate_new = validate
    try:
        report = base.collect(manifest, root, [], references, score)
    finally:
        base.validate_new = original
    report["fixed_final_base_steps"] = dict(manifest["fixed_final_base_steps"])
    return report


def combine(development_manifest, development, holdout_manifest, holdout):
    choices = development_choices(development_manifest, development)
    if (holdout["mode"] != "holdout" or holdout["expected_per_model"] != 800 or
        holdout["errors"] or holdout["invalid"] or
        holdout["fixed_final_base_steps"] != holdout_manifest["fixed_final_base_steps"]):
        raise ValueError("The fixed-step holdout report is incompatible")
    if any(not choices[model]["eligible"] or choices[model]["fixed_final_base_step"] != step
           for model, step in holdout_manifest["fixed_final_base_steps"].items()):
        raise ValueError("Holdout step was not selected by the complete development run")
    models = {}
    for model, step in holdout_manifest["fixed_final_base_steps"].items():
        rows = []
        for row in development["models"][model]["rows"]:
            rows.append(dict(case=row["case"], task=row["task"], length=row["length"],
                whole=row["whole"], zero=row["zero"], teacher=row["teacher"],
                cartridge=row["step" + str(step)], linked8=None, cartridge_percent=None))
        models[model] = dict(rows=rows, overall=summarize(rows))
    development_standard = dict(mode="development", expected_per_model=200,
        metric=development["metric"], selection_policy="decision_entity_link_v1",
        student_rollout_tokens=32, selected_transfer_weight=0., cached_scores=True,
        online_api_calls=0, evaluation_answers_used_only_posthoc=True,
        models=models, errors=[], invalid=[])
    result = base.combine_reports(development_standard, holdout)
    if result is not None:
        result["fixed_final_base_steps"] = dict(holdout_manifest["fixed_final_base_steps"])
        result["development_choices"] = choices
    return result


def render(report):
    lines = ["# Fixed-step Query Cartridge evaluation", "",
        "| Model | Cases | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher | SD steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, item in report["models"].items():
        summary = item["overall"]
        accuracy = summary["accuracy"]
        fmt = lambda value: "—" if value is None else f"{value:.2f}"
        lines.append(f"| {model} | {summary['n']} | {fmt(accuracy['whole'])} | {fmt(accuracy['zero'])} | "
            f"{fmt(accuracy['teacher'])} | {fmt(accuracy['cartridge'])} | "
            f"{fmt(summary['versus_teacher']['delta_pp'])} | {report['fixed_final_base_steps'][model]} |")
    if report["mode"] == "combined":
        lines += ["", "| Model / question type | Cases | Whole | Selected zero | ICL Teacher | Cartridge | vs Teacher |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for model, item in report["models"].items():
            for task in ("qa" + str(index) for index in range(1, 11)):
                summary = item["groups"].get("task:" + task)
                if not summary:
                    continue
                accuracy = summary["accuracy"]
                fmt = lambda value: "—" if value is None else f"{value:.2f}"
                lines.append(f"| {model} {task} | {summary['n']} | {fmt(accuracy['whole'])} | "
                    f"{fmt(accuracy['zero'])} | {fmt(accuracy['teacher'])} | "
                    f"{fmt(accuracy['cartridge'])} | {fmt(summary['versus_teacher']['delta_pp'])} |")
        for model, item in report["models"].items():
            summary = item["overall"]
            teacher, zero = summary["versus_teacher"], summary["versus_zero"]
            lines.append(f"\n{model}: 792-case source-disjoint target — {item['decision']['reason']}.")
            lines.append(f"Compared with Teacher, {teacher['new_correct']} new correct / "
                         f"{teacher['new_wrong']} new wrong; compared with selected zero, "
                         f"{zero['new_correct']} new correct / {zero['new_wrong']} new wrong.")
            size = item["holdout"]["median_cartridge_percent"]
            if size is not None:
                lines.append(f"Holdout median standalone document KV: {size:.2f}% of whole document KV "
                             "(excluding model weights, suffix KV and training peak).")
    lines += ["", "All methods use the same loose task-aware lexical score. The 200-case development "
        "run chooses the model-specific step before the 800 source-disjoint cases are generated. "
        "The original Teacher has an additional generic response-format instruction; the paired "
        "selected-zero and Cartridge use the same original question format.",
        "Selection reserves part of the 10% chunk budget for entity-linked context derived from "
        "cached Decision scores; the Student rollout cap is 32 tokens. The original OLMo profile "
        "used score-ranked chunks and an 8-token rollout, so this is an SD variant rather than "
        "an exact configuration reproduction.",
        "The final holdout output is a physically independent selected-KV cartridge, after releasing "
        "the full and source KV and request-local training state. No extra transfer loss or online API "
        "call is used.",
        "The 200 development checkpoint outputs are paired selected-KV diagnostics, not a serving "
        "memory measurement; 2/4-step checkpoints are validated against physical standalone outputs "
        "in model preflight. Two previously inspected source questions are excluded from the 792-case "
        "target. Aggregate scores from earlier runs have been seen, so this is not fully blind.", ""]
    return "\n".join(lines)


def write(report, output):
    for suffix, text in ((".json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                         (".md", render(report))):
        path = Path(output).with_suffix(suffix)
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
        raise ValueError("Unexpected shared scorer")
    report = collect(json.loads(args.manifest.read_text()), args.candidate_root,
                     json.loads(args.references.read_text()), score_prediction)
    write(report, args.output)


if __name__ == "__main__":
    main()
