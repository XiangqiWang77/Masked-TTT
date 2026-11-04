#!/usr/bin/env python3
"""Cross-tab completed Qwen answers with post-hoc annotated support coverage.

No selection, inference, or model training is run here. The support facts and
references are read only after the saved answers have already been produced.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

STATUSES = ("complete", "incomplete", "ambiguous", "unavailable")
OUTCOMES = ("both_correct", "full_only", "teacher_only", "both_wrong")
ARMS = ("whole", "zero", "teacher", "cartridge", "zero_teacher_format", "cartridge_teacher_format")


def outcome(row):
    if row["whole"]:
        return "both_correct" if row["teacher"] else "full_only"
    return "teacher_only" if row["teacher"] else "both_wrong"


def summarize(rows):
    if any(row["coverage"] not in STATUSES or
           any(type(row.get(key)) is not int or row[key] not in (0, 1) for key in ARMS)
           for row in rows):
        raise ValueError("Invalid coverage or answer score")
    n = len(rows)
    pairs = {}
    for old in ("whole", "zero", "teacher"):
        gained = sum(row["cartridge"] and not row[old] for row in rows)
        lost = sum(row[old] and not row["cartridge"] for row in rows)
        pairs[old] = dict(new_correct=gained, new_wrong=lost,
                          delta_pp=100*(gained-lost)/n if n else None)
    return dict(n=n, source_questions=len({row["source"] for row in rows}),
        accuracy={key: 100*sum(row[key] for row in rows)/n if n else None for key in ARMS},
        correct={key: sum(row[key] for row in rows) for key in ARMS}, pairs=pairs,
        coverage_counts={key: sum(row["coverage"] == key for row in rows) for key in STATUSES},
        answer_groups={key: sum(outcome(row) == key for row in rows) for key in OUTCOMES})


def groups(rows):
    result = {}
    for name in STATUSES:
        subset = [row for row in rows if row["coverage"] == name]
        result["coverage:"+name] = summarize(subset)
    for name in OUTCOMES:
        subset = [row for row in rows if outcome(row) == name]
        result["full_teacher:"+name] = summarize(subset)
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        result["task:"+task] = summarize(task_rows)
        result["task:"+task+"/full_only"] = summarize([row for row in task_rows if outcome(row) == "full_only"])
    for length in sorted({row["length"] for row in rows}):
        length_rows = [row for row in rows if row["length"] == length]
        result["length:"+str(length)] = summarize(length_rows)
        result["length:"+str(length)+"/full_only"] = summarize([row for row in length_rows if outcome(row) == "full_only"])
    return result


def collect(workspace, snapshot):
    dev = workspace / "query_cartridges_linked_selection_20261104"
    held = workspace / "query_cartridges_linked_holdout_20261104"
    development = json.loads((dev / "manifest.json").read_text())
    holdout = json.loads((held / "manifest.json").read_text())
    dev_cases, held_cases = development["case_ids"], holdout["case_ids"]
    if (len(dev_cases) != 200 or len(set(dev_cases)) != 200 or
        len(held_cases) != 800 or len(set(held_cases)) != 800 or set(dev_cases) & set(held_cases) or
        set(development["holdout_case_ids"]) != set(held_cases) or
        snapshot["metric"] != "answer_focused_lexical_v2" or snapshot["selected_fraction"] != .1 or
        snapshot["selection_policy"] != "decision_entity_link_v1" or
        snapshot["models"]["qwen"]["state"] != "complete" or
        snapshot["models"]["qwen"]["combined"]["n"] != 1000):
        raise ValueError("The completed Qwen source split or frozen report changed")

    protocol = json.loads(Path(development["model_protocols"]["qwen"]).read_text())
    if protocol != json.loads(Path(holdout["model_protocols"]["qwen"]).read_text()):
        raise ValueError("Qwen inference protocol changed across splits")
    index = {entry["key"]: entry for entry in json.loads(Path(protocol["index"]).read_text())}
    score_root = Path(protocol["shared_scores"])
    score_files = json.loads((score_root / "case_to_score_file.json").read_text())
    previous = {row["case"]: row for row in json.loads((dev / "coverage.json").read_text())["rows"]}
    if set(previous) != set(dev_cases):
        raise ValueError("The prior 200-case annotated coverage changed")

    helper_file = workspace / "query_cartridges_20261103/audit_babilong_selected_support.py"
    spec = importlib.util.spec_from_file_location("saved_support_audit", helper_file)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    panel = {row["case"]: row for row in json.loads((support.ORIGINAL / "fresh_confirmation.json").read_text())}
    units, matches = support.source_matches(json.loads(support.RESERVATION.read_text()))
    references = json.loads((workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json").read_text())
    sys.path.insert(0, str(workspace / "monitor"))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected unified answer scorer")

    roots = {case: dev for case in dev_cases}
    roots.update({case: held for case in held_cases})
    cases = list(roots)
    if not set(cases).issubset(index.keys() & panel.keys() & score_files.keys() & references.keys()):
        raise ValueError("A completed question is missing its source, scores, or reference")
    rows, checked_development = [], 0
    with open(protocol["source_cases"], "rb") as source:
        for key in sorted(cases, key=lambda item: index[item]["offset"]):
            entry, root = index[key], roots[key]
            source.seek(entry["offset"])
            case = json.loads(source.read(entry["bytes"]))
            stored = json.loads((root / "cases/qwen" / key / "result.json").read_text())
            selection = stored["selection"]
            unit = panel[key]["sampling_unit"]
            if (case["key"] != key or case["query"].strip() != units[unit]["question"].strip() or
                stored["key"] != key or stored["task"] != references[key]["task"] or
                stored["model"] != "Qwen/Qwen3-4B-Instruct-2507" or
                selection["source_case_key"] != key or not selection["complete_original_document"] or
                selection["source_mapping"] is not None or selection["selection_policy"] != "decision_entity_link_v1" or
                selection["entity_reserve_fraction"] != .25 or selection["online_api_calls"] != 0 or
                selection["reference_answers_used"] or stored["training"]["selected_transfer_weight"] != 0):
                raise ValueError("The original completed Qwen case changed")
            selected_groups = selection["selected_original_chunks"]
            if any(len(group) != 1 for group in selected_groups):
                raise ValueError("Unexpected original chunk mapping")
            selected = sorted(group[0] for group in selected_groups)
            if (len(set(selected)) != len(selected) or
                len(selected) != max(1, math.ceil(selection["total_chunks"]*.1))):
                raise ValueError("The 10% selected-chunk budget changed")
            scored = json.loads((score_root / "scores" / score_files[key]).read_text())
            spans = scored["source_spans"]
            if (scored["case_key"] != key or scored["question"] != case["query"] or
                scored["reference_answers_used"] is not False or len(spans) != selection["total_chunks"] or
                any(not 0 <= a < b <= len(case["document"]) for a,b in spans) or
                any(i < 0 or i >= len(spans) for i in selected)):
                raise ValueError("The cached source chunk identity changed")
            ranges = [tuple(spans[i]) for i in selected]
            measured = support.combine([support.support_coverage(case["document"], match, ranges,
                strip_boundary_punctuation=True) for match in matches[unit]])
            if measured["status"] not in STATUSES:
                raise ValueError("Unknown support coverage status")
            if key in previous:
                before = previous[key]["outcomes"]["entity10"]
                if before["selected"] != selected or before["status"] != measured["status"]:
                    raise ValueError("Recomputed support differs from the previously checked development case")
                checked_development += 1
            pair = stored["standalone_cartridge"]["paired_prompt_control"]
            if (not pair["same_selected_kv"] or not pair["same_generation_limit"] or
                pair["reference_answers_used"] or pair["teacher_answer_in_student_prompt"]):
                raise ValueError("The saved outputs do not share the selected cache")
            reference = references[key]
            grade = lambda text: int(score_prediction(text, reference["answers"], "babilong", reference["task"])[0])
            rows.append(dict(source=key.split("-",2)[2], task=reference["task"], length=entry["length_bucket"],
                split="development" if root == dev else "holdout", coverage=measured["status"],
                whole=grade(stored["self_judge"]["native_prediction"]), teacher=grade(stored["teacher"]["prediction"]),
                zero=grade(pair["zero_original"]["prediction"]), cartridge=grade(pair["trained_original"]["prediction"]),
                zero_teacher_format=grade(pair["zero_teacher_suffix"]["prediction"]),
                cartridge_teacher_format=grade(pair["trained_teacher_suffix"]["prediction"])))
    if len(rows) != 1000 or checked_development != 200 or len({row["source"] for row in rows}) != 250:
        raise ValueError("The 1,000 cases, 250 sources, or prior coverage checks are incomplete")
    overall = summarize(rows)
    frozen = snapshot["models"]["qwen"]["combined"]["accuracy"]
    if any(not math.isclose(overall["accuracy"][arm], frozen[arm], abs_tol=1e-9) for arm in ARMS):
        raise ValueError("Rescored outcomes do not match the completed 1,000-case snapshot")
    return dict(metric="answer_focused_lexical_v2", cases=1000, source_questions=250,
        selected_fraction=.1, selection_policy="decision_entity_link_v1",
        answers_and_support_annotations_used_only_posthoc=True, prior_development_coverage_checks=checked_development,
        no_new_model_calls=True, online_api_calls=0,
        support_complete_means_all_annotated_fact_spans_covered=True,
        coverage_does_not_prove_use_or_complete_all_information=True,
        overall=overall, groups=groups(rows),
        splits={name: dict(overall=summarize(subset),
                           full_only=summarize([row for row in subset if outcome(row) == "full_only"]))
                for name in ("development", "holdout")
                if (subset := [row for row in rows if row["split"] == name])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.workspace.resolve(), json.loads(args.snapshot.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("overall", json.dumps(result["overall"]))
    print("full_only", json.dumps(result["groups"]["full_teacher:full_only"]))


if __name__ == "__main__":
    main()
