#!/usr/bin/env python3
"""Audit paired Value-basis development results after model generation."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(rows):
    n = len(rows)
    names = ("whole", "zero", "teacher", "document", "selected")
    acc = {name: 100 * sum(row[name] for row in rows) / n if n else None for name in names}
    gained = sum(not row["document"] and row["selected"] for row in rows)
    lost = sum(row["document"] and not row["selected"] for row in rows)
    return dict(n=n, accuracy=acc, selected_vs_document=dict(new_correct=gained, new_wrong=lost,
                delta_pp=100 * (gained - lost) / n if n else None),
        median_document_basis_energy=statistics.median([r["document_basis_energy"] for r in rows]) if n else None,
        median_selected_basis_energy=statistics.median([r["selected_basis_energy"] for r in rows]) if n else None)


def audit_pair(key, row, document, selected, reference, grade):
    for result, mode in ((document, "document"), (selected, "selected")):
        t, s, pair = result["training"], result["selection"], result["standalone_cartridge"]["paired_prompt_control"]
        probe = result["basis_probe"]
        release = result["memory"]["full_cache_release_before_standalone"]
        if (result["key"] != key or result["basis_source"] != mode or t["value_basis_source"] != mode or
            t["objective"] != "on-policy KL(teacher || student)" or t["selected_transfer_weight"] != 0 or
            any("selected_transfer_KL_before" in step for step in t["steps"]) or not t["steps"] or
            not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
            s["source_case_key"] != key or s["reference_answers_used"] or s["online_api_calls"] != 0 or
            not s["complete_original_document"] or not release["requested"] or
            release["full_cache_tensors_live_before_standalone"] != 0 or
            result["self_study"]["reference_answers_used"] or result["self_study"]["student_answer_hint_used"] or
            result["teacher_answer_in_final_prompt"] or pair["teacher_answer_in_student_prompt"] or
            pair["reference_answers_used"] or not pair["same_selected_kv"] or not pair["same_generation_limit"] or
            not pair["same_backbone_and_device"] or pair["control_update_steps"] != 0 or
            probe["model_or_cache_modified"] or probe["reference_answers_used"] or probe["rank"] != 8):
            raise ValueError("The paired basis experiment contract changed")
    pa, pb = (item["standalone_cartridge"]["paired_prompt_control"] for item in (document, selected))
    if (document["model"] != selected["model"] or any(document[k] != selected[k] for k in ("job", "gpu", "node")) or
        document["length"] != selected["length"] or document["length"] != row["length"] or
        any(document["basis_probe"][k] != selected["basis_probe"][k] for k in ("selected_tokens", "sampled_tokens")) or
        document["task"] != selected["task"] or
        document["task"] != row["task"] or reference["task"] != row["task"] or
        document["selection"]["selected_original_chunks"] != selected["selection"]["selected_original_chunks"] or
        document["teacher"]["tokens"] != selected["teacher"]["tokens"] or
        pa["zero_original"]["tokens"] != pb["zero_original"]["tokens"] or
        pa["zero_teacher_suffix"]["tokens"] != pb["zero_teacher_suffix"]["tokens"] or
        document["self_judge"]["native_prediction"] != selected["self_judge"]["native_prediction"]):
        raise ValueError("The basis arms do not share the same evidence and controls")
    def score(text):
        value = grade(text, reference["answers"], "babilong", reference["task"])[0]
        if type(value) not in (int, float) or value not in (0, 1):
            raise ValueError("The common judge must be binary")
        return int(value)
    values = dict(whole=score(document["self_judge"]["native_prediction"]),
        zero=score(pa["zero_original"]["prediction"]), teacher=score(document["teacher"]["prediction"]),
        document=score(pa["trained_original"]["prediction"]), selected=score(pb["trained_original"]["prediction"]))
    for mode, result in (("document", document), ("selected", selected)):
        value = result["basis_probe"]["mean_selected_energy_full_basis"]
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid basis energy")
        values[mode + "_basis_energy"] = value
    if any(row.get(name) != value for name, value in values.items()):
        raise ValueError("The saved score or basis measurement changed")
    return dict(case=key, task=row["task"], length=row["length"], **values)


def audit(manifest, report, root, references, grade):
    keys = manifest["case_ids"]
    if (manifest["case_count"] != 40 or len(keys) != 40 or len(set(keys)) != 40 or
        manifest["basis_sources"] != ["document", "selected"] or manifest["selected_transfer_weight"] != 0 or
        report["metric"] != "answer_focused_lexical_v2" or report["development_only"] is not True or
        report["independent_source_questions"] != 10 or report["variants"] != ["document", "selected"] or
        report["same_student_teacher_objective"] is not True or
        report["reference_answers_used_only_posthoc"] is not True or report["online_api_calls"] != 0 or
        report["errors"] or report["invalid"]):
        raise ValueError("The fixed 40-case basis study changed")
    sources = defaultdict(set)
    for key in keys:
        pieces = key.split("-", 2)
        if len(pieces) != 3 or pieces[0] != "babilong" or pieces[1] not in ("8", "16", "32", "64"):
            raise ValueError("Unexpected source case")
        sources[pieces[2]].add(pieces[1])
    if len(sources) != 10 or any(lengths != {"8", "16", "32", "64"} for lengths in sources.values()):
        raise ValueError("Expected ten questions at four lengths")
    models = {}
    for model in ("qwen", "gpt"):
        saved = {row["case"]: row for row in report["models"][model]["rows"]}
        if len(report["models"][model]["rows"]) != 40 or set(saved) != set(keys):
            raise ValueError("Incomplete paired basis results")
        rows = []
        for key in keys:
            folder = Path(root) / "cases" / model / key
            left = json.loads((folder / "document.json").read_text())
            right = json.loads((folder / "selected.json").read_text())
            rows.append(audit_pair(key, saved[key], left, right, references[key], grade))
        groups = defaultdict(list)
        for row in rows:
            groups["task:" + row["task"]].append(row)
            groups["length:" + str(row["length"])].append(row)
        models[model] = dict(overall=summarize(rows),
                             groups={name: summarize(items) for name, items in sorted(groups.items())})
    return dict(scope="40-case development; ten questions at four lengths",
        metric=report["metric"], variants=report["variants"], rank=8, online_api_calls=0, new_model_calls=0,
        reference_answers_used_only_after_generation=True,
        teacher_has_extra_generic_format_instruction=True, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "report", "case-root", "references", "scorer-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_root))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected scorer")
    data = audit(json.loads(args.manifest.read_text()), json.loads(args.report.read_text()), args.case_root,
                 json.loads(args.references.read_text()), score_prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for model, info in data["models"].items():
        print(model, json.dumps(info["overall"]))


if __name__ == "__main__":
    main()
