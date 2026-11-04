#!/usr/bin/env python3
"""Aggregate support changes on the seven completed selection-change cases."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.question_overlap_selector import question_overlap
from masked_ttt.linked_selection import choose


def summarize(rows):
    transitions = Counter(f'{row["old_coverage"]}->{row["new_coverage"]}' for row in rows)
    result = {"cases": len(rows), "coverage_transitions": dict(sorted(transitions.items()))}
    for outcome in ("teacher", "cartridge", "zero"):
        gains = [row for row in rows if row[outcome] and not row["old_" + outcome]]
        losses = [row for row in rows if row["old_" + outcome] and not row[outcome]]
        result[outcome + "_new_correct"] = len(gains)
        result[outcome + "_new_wrong"] = len(losses)
        result[outcome + "_gains_by_coverage_transition"] = dict(sorted(Counter(
            f'{row["old_coverage"]}->{row["new_coverage"]}' for row in gains).items()))
    return result


def collect(workspace):
    baseline_root = workspace / "query_cartridges_linked_selection_20261104"
    candidate_root = workspace / "query_cartridges_proper_where_qwen_dev_20261104"
    report = json.loads((candidate_root / "latest.json").read_text())
    if (report["model"] != "qwen" or report["mode"] != "development" or report["expected"] != 200 or
        report["metric"] != "answer_focused_lexical_v2" or len(report["rows"]) != 200 or
        report["errors"] or report["invalid"] or report["unmatched"]):
        raise ValueError("Expected a completed Qwen 200-case development report")
    changed = [row for row in report["rows"] if row["selection_changed"]]
    if len(changed) != 7:
        raise ValueError("Unexpected number of selection-change cases")
    manifest = json.loads((baseline_root / "manifest.json").read_text())
    protocol = json.loads(Path(manifest["model_protocols"]["qwen"]).read_text())
    index = {entry["key"]: entry for entry in json.loads(Path(protocol["index"]).read_text())}
    root_scores = Path(protocol["shared_scores"])
    score_files = json.loads((root_scores / "case_to_score_file.json").read_text())
    spec = importlib.util.spec_from_file_location("saved_support_audit", workspace / "query_cartridges_20261103/audit_babilong_selected_support.py")
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    panel = {row["case"]: row for row in json.loads((support.ORIGINAL / "fresh_confirmation.json").read_text())}
    units, matches = support.source_matches(json.loads(support.RESERVATION.read_text()))
    rows = []
    with open(protocol["source_cases"], "rb") as source:
        for old in changed:
            case_key = old["case"]
            entry = index[case_key]
            source.seek(entry["offset"])
            case = json.loads(source.read(entry["bytes"]))
            scored = json.loads((root_scores / "scores" / score_files[case_key]).read_text())
            if (case["key"] != case_key or scored["case_key"] != case_key or
                scored["question"] != case["query"] or scored["reference_answers_used"] is not False):
                raise ValueError("Document, question or cached scores do not match")
            spans, records = scored["source_spans"], scored["records"]
            # Fix both selections before accessing source annotations.
            old_ids = choose(case["document"], case["query"], spans, records, .1, "entity", .25)
            new_ids = question_overlap(case["document"], case["query"], spans, records, .1, mode="proper_where")
            old_result = json.loads((baseline_root / "cases/qwen" / case_key / "result.json").read_text())
            new_result = json.loads((candidate_root / "cases/qwen" / case_key / "result.json").read_text())
            saved = lambda result: sorted(i for group in result["selection"]["selected_original_chunks"] for i in group)
            if (old_ids != saved(old_result) or new_ids != saved(new_result) or old_ids == new_ids or
                len(old_ids) != len(new_ids)):
                raise ValueError("Reconstructed selections differ from the saved same-budget runs")
            unit = panel[case_key]["sampling_unit"]
            if case["query"].strip() != units[unit]["question"].strip():
                raise ValueError("Question and supporting-fact source differ")
            def coverage(ids):
                ranges = [tuple(spans[i]) for i in ids]
                return support.combine([support.support_coverage(case["document"], match, ranges,
                    strip_boundary_punctuation=True) for match in matches[unit]])["status"]
            rows.append(dict(old_coverage=coverage(old_ids), new_coverage=coverage(new_ids),
                **{key: old[key] for key in ("teacher", "old_teacher", "zero", "old_zero", "cartridge", "old_cartridge")}))
    return dict(summary=summarize(rows), selection_before_annotation=True,
                source_annotations_used_for_selection=False, new_model_calls=0, online_api_calls=0,
                development_selection_designed_after_reviewing_development_results=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
