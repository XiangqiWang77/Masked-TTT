#!/usr/bin/env python3
"""Score matched 2/4/8-step cartridge checkpoints after inference."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_report import validate_new


def validate_result(result, case, model, task):
    validate_new(case, result, model, task)
    training = result["training"]
    probe = result["standalone_cartridge"]["step_probe"]
    practices = result["self_study"]["verified"]
    if (not isinstance(result["self_judge"]["native_prediction"], str) or
        training["gain_head_mode"] != "kv" or len(training["steps"]) != 8 + practices or
        set(probe["checkpoints"]) != {"2", "4", "8"} or probe["practice_count"] != practices or
        any(probe["checkpoints"][str(n)]["total_steps"] != n + practices for n in (2, 4, 8)) or
        probe["checkpoints"]["8"]["tokens"] != result["generated_token_ids"] or
        not probe["same_selected_kv"] or probe["full_kv_used_at_decode"] or
        probe["reference_answers_used"] or probe["teacher_answer_in_final_prompt"] or
        not probe["diagnostic_snapshots_offloaded_to_cpu"] or
        not probe["diagnostic_snapshots_released_before_final_decode"] or
        (model == "gpt" and result["request_chat_date"] != "2026-11-03")):
        raise ValueError("Invalid step checkpoint or physical-release record")


def summarize(rows):
    keys = ("whole", "zero", "teacher", "step2", "step4", "step8")
    if any(type(row[key]) is not int or row[key] not in (0, 1) for row in rows for key in keys):
        raise ValueError("Expected binary post-generation scores")
    n = len(rows)
    def paired(new, old):
        correct = sum(row[new] and not row[old] for row in rows)
        wrong = sum(row[old] and not row[new] for row in rows)
        return dict(new_correct=correct, new_wrong=wrong,
                    delta_pp=100 * (correct - wrong) / n if n else None)
    return dict(n=n, source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
        accuracy={key: 100 * sum(row[key] for row in rows) / n if n else None for key in keys},
        step4_vs_step2=paired("step4", "step2"), step8_vs_step2=paired("step8", "step2"),
        step8_vs_teacher=paired("step8", "teacher"), step8_vs_zero=paired("step8", "zero"))


def collect(manifest, root, references, score):
    if (manifest["case_count"] != 56 or len(manifest["case_ids"]) != 56 or
        len(set(manifest["case_ids"])) != 56 or manifest["step_budgets"] != [2, 4, 8] or
        manifest["student_rollout_tokens"] != 32 or manifest["selected_transfer_weight"] != 0 or
        manifest["selection_policy"] != "decision_entity_link_v1" or manifest["keep_fraction"] != .1 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The frozen pilot protocol changed")
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
            checkpoint = result["standalone_cartridge"]["step_probe"]["checkpoints"]
            controls = result["standalone_cartridge"]["original_question_controls"]
            rows.append(dict(case=case, task=ref["task"], length=result["length"],
                whole=grade(result["self_judge"]["native_prediction"]),
                zero=grade(controls["zero"]["prediction"]), teacher=grade(result["teacher"]["prediction"]),
                step2=grade(checkpoint["2"]["prediction"]), step4=grade(checkpoint["4"]["prediction"]),
                step8=grade(checkpoint["8"]["prediction"])))
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
    lines = ["# Longer on-policy rollout: matched SD step checkpoints", "",
        "| Model / type | n | Whole | Selected zero | ICL Teacher | 2 steps | 4 steps | 8 steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, item in report["models"].items():
        groups = [(model, item["overall"])]+[(model+" "+name[5:], stats)
                  for name, stats in sorted(item["groups"].items()) if name.startswith("task:")]
        for name, stats in groups:
            acc = stats["accuracy"]
            lines.append(f"| {name} | {stats['n']} | " + " | ".join(fmt(acc[key])
                for key in ("whole", "zero", "teacher", "step2", "step4", "step8")) + " |")
    lines += ["", "Checkpoints share one on-policy teacher, one selected KV and one optimization trajectory; "
              "each base budget also includes one update per accepted practice question. The 2/4-step outputs "
              "are matched selected-KV diagnostics; the formal 8-step output is decoded after source KV and "
              "training state are released. All reference answers are used only for post-generation scoring.",
              "", "The 56-case pilot includes all five QA3 development sources and one fixed source from each "
              "other type, each at four lengths. It is a focused development experiment, not a 1,000-case result.", ""]
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
