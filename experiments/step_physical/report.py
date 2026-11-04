#!/usr/bin/env python3
"""Score physical 8-step holdout and matched checkpoints after generation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.long_rollout_steps.report import summarize
from experiments.step_physical.protocol import development_choice
from experiments.value_lr_physical.report import validate_result

INSPECTED = {"qa1-300023", "qa2-300037"}


def score_rows(manifest, root, references, score):
    if (manifest["mode"] != "holdout" or manifest["case_count"] != 800 or manifest["models"] != ["gpt"] or
        manifest["selection_policy"] != "cached_decision_score10" or manifest["steps"] != 8 or
        manifest["fixed_value_learning_rate"] != .0001 or manifest["student_rollout_tokens"] != 8 or
        manifest["selected_transfer_weight"] != 0 or manifest["reference_answers_used_in_inference"] is not False or
        manifest["online_api_calls"] != 0 or manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The development-fixed original-selector step holdout changed")
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    if set(manifest["case_ids"]) != set(manifest["holdout_case_ids"]):
        raise ValueError("The source-disjoint holdout cases changed")
    rows, errors, invalid = [], [], []
    for case in manifest["case_ids"]:
        folder = Path(root) / "cases/gpt" / case
        path = folder / "result.json"
        if not path.exists():
            if (folder / "error.json").exists():
                error = json.loads((folder / "error.json").read_text())
                errors.append(dict(case=case, type=error.get("type")))
            continue
        result, ref = json.loads(path.read_text()), references[case]
        try:
            validate_result(result, case, "gpt", ref["task"], .0001, base_steps=8)
            probe = result["standalone_cartridge"]["step_probe"]
            if (result["training"]["fixed_base_steps"] != 8 or probe["fixed_final_base_step"] != 8 or
                set(probe["checkpoints"]) != {"2", "4", "8"} or
                probe["checkpoints"]["8"]["tokens"] != result["generated_token_ids"] or
                not probe["same_selected_kv"] or probe["full_kv_used_at_decode"] or
                probe["reference_answers_used"] or probe["teacher_answer_in_final_prompt"] or
                not probe["diagnostic_snapshots_offloaded_to_cpu"] or
                not probe["diagnostic_snapshots_released_before_final_decode"]):
                raise ValueError("The fixed physical checkpoint record changed")
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(dict(case=case, reason=str(exc)))
            continue
        grade = lambda text: int(score(text, ref["answers"], "babilong", ref["task"])[0])
        controls = result["standalone_cartridge"]["original_question_controls"]
        checkpoint = probe["checkpoints"]
        full = result["memory"]["full_document_kv_bytes"]
        card = result["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]
        if not full or not card:
            raise ValueError("Missing materialized cache size")
        row = dict(case=case, task=result["task"], length=result["length"],
            whole=grade(result["self_judge"]["native_prediction"]), zero=grade(controls["zero"]["prediction"]),
            teacher=grade(result["teacher"]["prediction"]),
            **{"step" + str(step): grade(checkpoint[str(step)]["prediction"]) for step in (2, 4, 8)},
            cartridge_percent=100 * card / full)
        row["cartridge"] = row["step8"]
        rows.append(row)
    return rows, errors, invalid


def result(mode, manifest, rows, errors=(), invalid=()):
    expected = 800 if mode == "holdout" else 1000
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    strict = [row for row in rows if row["case"].split("-", 2)[2] not in INSPECTED]
    stats = summarize(strict)
    passed = None
    if mode == "holdout" and stats["n"] == 792:
        accuracy = stats["accuracy"]
        passed = (accuracy["step8"] - accuracy["teacher"] >= 3 - 1e-9 and
                  all(accuracy["step8"] > accuracy[key] for key in ("whole", "zero", "step2")))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=mode, model="gpt", expected=expected,
        metric="answer_focused_lexical_v2", fixed_base_steps=8, fixed_value_learning_rate=.0001,
        cached_scores=True, online_api_calls=0, reference_answers_used_only_posthoc=True,
        overall=summarize(rows), groups={key: summarize(value) for key, value in groups.items()},
        strict_holdout=summarize(strict) if mode == "holdout" else None,
        strict_target_passed=passed, rows=rows, errors=list(errors), invalid=list(invalid))


def collect(manifest, root, references, score):
    return result("holdout", manifest, *score_rows(manifest, root, references, score))


def combine(development_manifest, development, manifest, holdout):
    choice = development_choice(development_manifest, development, "gpt")
    if (choice["steps"] != 8 or manifest["steps"] != 8 or holdout["fixed_base_steps"] != 8 or
        holdout["errors"] or holdout["invalid"]):
        raise ValueError("The holdout no longer matches the development-fixed 8-step choice")
    dev = [dict(row, cartridge=row["step8"], cartridge_percent=None)
           for row in development["models"]["gpt"]["rows"]]
    held = holdout["rows"]
    if len(dev) != 200 or len(held) != 800:
        return None
    validate_partition([row["case"] for row in dev], [row["case"] for row in held])
    combined = result("combined", manifest, dev + held)
    combined.update(development=summarize(dev), holdout=holdout["overall"],
                    strict_holdout=holdout["strict_holdout"], strict_target_passed=holdout["strict_target_passed"])
    return combined


def render(report):
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    groups = [("All", report["overall"])] + [(key[5:], value) for key, value in sorted(report["groups"].items())
                                                if key.startswith("task:")]
    groups += [(str(int(key[7:]) // 1024) + "K", value) for key, value in sorted(report["groups"].items())
               if key.startswith("length:")]
    lines = ["# GPT-OSS: physical original-selector 8-step cartridge", "",
        "| Group | n | Whole | Selected zero | ICL Teacher | 2 steps | 4 steps | 8 steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, item in groups:
        accuracy = item["accuracy"]
        lines.append(f"| {label} | {item['n']} | " + " | ".join(
            fmt(accuracy[name]) for name in ("whole", "zero", "teacher", "step2", "step4", "step8")) + " |")
    stats = report["overall"]
    lines += ["", f"8 vs Teacher: {stats['step8_vs_teacher']['new_correct']} new correct / "
        f"{stats['step8_vs_teacher']['new_wrong']} new wrong; 8 vs zero: "
        f"{stats['step8_vs_zero']['new_correct']} / {stats['step8_vs_zero']['new_wrong']}.",
        f"8 vs 2: {stats['step8_vs_step2']['new_correct']} new correct / "
        f"{stats['step8_vs_step2']['new_wrong']} new wrong. Strict 792-case target: {report['strict_target_passed']}.",
        "The selected chunk set is the original cached Decision-score top 10%; SD uses an 8-token student rollout "
        "and Value learning rate 0.0001. Final 8-step answers use the physically independent KV-only cartridge.",
        "The Teacher has an extra generic format instruction; selected-zero and Cartridge use the same original question. "
        "All answers use the same task-aware loose lexical score. The step was fixed from 200 development cases; "
        "earlier aggregate results have been viewed, so the source-disjoint holdout is not fully blind.", ""]
    return "\n".join(lines)


def write(report, stem):
    for suffix, text in ((".json", json.dumps(report, indent=2) + "\n"), (".md", render(report))):
        path = Path(stem).with_suffix(suffix)
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
