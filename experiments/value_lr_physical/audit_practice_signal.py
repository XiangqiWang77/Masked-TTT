#!/usr/bin/env python3
"""Compare logged practice coverage with paired outcomes across backbones.

This is post-generation, descriptive analysis of already completed runs.
It never exposes source text, answers or per-case identities in its output.
"""
import argparse
import json
import statistics
from pathlib import Path


def number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Expected a numeric training statistic")
    return float(value)


def summarize(rows):
    if not rows:
        return {"n": 0}
    result = {"n": len(rows), "verified_practice_cases": sum(r["verified"] > 0 for r in rows),
              "median_first_kl": statistics.median(r["first_kl"] for r in rows),
              "median_first_gain_gradient": statistics.median(r["gain_grad"] for r in rows),
              "median_first_value_gradient": statistics.median(r["value_grad"] for r in rows),
              "mean_update_steps": statistics.mean(r["steps"] for r in rows)}
    for name in ("whole", "teacher", "selective", "cartridge", "selective_teacher_format", "cartridge_teacher_format"):
        values = [r[name] for r in rows if r.get(name) is not None]
        if values:
            if len(values) != len(rows) or any(v not in (0, 1) for v in values):
                raise ValueError("Paired outcome is missing or is not binary")
            result[name + "_accuracy"] = 100 * sum(values) / len(values)
    for base in ("teacher", "selective", "selective_teacher_format"):
        candidate = "cartridge_teacher_format" if base == "selective_teacher_format" else "cartridge"
        pairs = [(r[base], r[candidate]) for r in rows if r.get(base) is not None and r.get(candidate) is not None]
        if pairs:
            if len(pairs) != len(rows):
                raise ValueError("Partial paired outcomes")
            gains = sum(a == 0 and b == 1 for a, b in pairs)
            losses = sum(a == 1 and b == 0 for a, b in pairs)
            result["vs_" + base] = dict(new_correct=gains, new_wrong=losses,
                                        delta_points=100 * (gains - losses) / len(pairs))
    for name in ("original_tokens_changed", "teacher_format_tokens_changed"):
        values = [r[name] for r in rows if r.get(name) is not None]
        if values:
            if len(values) != len(rows) or any(type(v) is not bool for v in values):
                raise ValueError("The matched-output control is incomplete")
            result[name] = sum(values)
    return result


def load_model(workspace, model):
    if model == "olmo":
        root = workspace / "query_cartridges_20261103"
        report = json.loads((root / "latest.json").read_text())
        entry = next(x for x in report["jobs"] if x["name"] == "olmo3-7b-babilong")
        if (entry["all_paired_n"] != 1000 or report["metric"] !=
            "answer-focused task-aware lexical v2 / format-tolerant choice accuracy"):
            raise ValueError("The complete original OLMo paired report is required")
        source = []
        for x in entry["case_outcomes"]:
            s = x["scores"]
            source.append(dict(case=x["case"], task=x["task"], whole=s["vanilla"],
                               teacher=s["icl_teacher"], cartridge=s["standalone_cartridge"]))
        cases = root / "runs/olmo3-7b-babilong/cases"
        filename = "masked_ttt.json"
    else:
        folders = {"qwen": "query_cartridges_qwen_olmo1000_20261104",
                   "gpt": "query_cartridges_gpt_olmo1000_20261103"}
        root = workspace / folders[model]
        report = json.loads((root / "latest.json").read_text())
        if report["completed"] != 1000 or report["metric"] != "answer_focused_lexical_v2":
            raise ValueError("The complete paired 1,000-case report is required")
        source = report["rows"]
        cases = root / "cases"
        filename = "weight_0.00.json"
    if len(source) != 1000 or len({x["case"] for x in source}) != 1000:
        raise ValueError("Expected 1,000 unique completed cases")
    rows = []
    for item in source:
        case = item["case"]
        log = json.loads((cases / case / filename).read_text())
        study, training = log["self_study"], log["training"]
        steps = training["steps"]
        if not steps or training["objective"] != "on-policy KL(teacher || student)":
            raise ValueError("Expected original on-policy teacher KL and nonempty updates")
        verified = study["verified"]
        if type(verified) is not int or verified not in (0, 1, 2):
            raise ValueError("Unexpected verified practice count")
        row = dict(task=item["task"], verified=verified, steps=len(steps),
                   first_kl=number(steps[0]["teacher_KL_before"]),
                   gain_grad=number(steps[0]["gain_gradient_norm"]),
                   value_grad=number(steps[0]["value_gradient_norm"]))
        control = log["standalone_cartridge"].get("paired_prompt_control")
        if control is not None:
            if (not control["same_selected_kv"] or not control["same_backbone_and_device"] or
                not control["same_generation_limit"] or control["control_update_steps"] != 0 or
                control["teacher_answer_in_student_prompt"] or control["reference_answers_used"]):
                raise ValueError("The zero-update answer comparison is not matched")
            for suffix, key in (("original", "original_tokens_changed"),
                                ("teacher_suffix", "teacher_format_tokens_changed")):
                before, after = control["zero_" + suffix]["tokens"], control["trained_" + suffix]["tokens"]
                if not isinstance(before, list) or not isinstance(after, list):
                    raise ValueError("Missing paired token sequences")
                row[key] = before != after
        for key in ("whole", "teacher", "selective", "cartridge", "selective_teacher_format", "cartridge_teacher_format"):
            row[key] = item.get(key)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = {"scope": "post-generation descriptive audit; no training, API or new model calls",
            "metric": "answer_focused_lexical_v2", "source_questions_per_model": 250,
            "same_sources_across_four_context_lengths": True,
            "teacher_and_original_cartridge_formats_differ": True,
            "models": {}}
    for name in ("olmo", "qwen", "gpt"):
        rows = load_model(args.workspace, name)
        data["models"][name] = dict(overall=summarize(rows),
            by_verified_practices={str(count): summarize([r for r in rows if r["verified"] == count])
                                   for count in (0, 1, 2)},
            by_task={task: summarize([r for r in rows if r["task"] == task])
                     for task in sorted({r["task"] for r in rows})})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for model, content in data["models"].items():
        print(model, json.dumps(content["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
