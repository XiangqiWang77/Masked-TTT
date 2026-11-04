#!/usr/bin/env python3
"""Compare label-free same-budget selectors on the fixed 200-case development set."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.babilong_linked_selection.question_overlap_selector import question_overlap
from masked_ttt.linked_selection import choose

VARIANTS = ("entity25", "mix", "lexical", "proper_where")


def summarize(rows, name):
    complete = lambda row, variant: row["coverage"][variant] == "complete"
    missing = lambda row, variant: row["coverage"][variant] == "incomplete"
    gained = sum(missing(row,"entity25") and complete(row,name) for row in rows)
    lost = sum(complete(row,"entity25") and missing(row,name) for row in rows)
    full_only = [row for row in rows if row["whole"] and not row["teacher"]]
    teacher_only = [row for row in rows if row["teacher"] and not row["whole"]]
    return dict(n=len(rows), selected_chunks=sum(row["selected_count"] for row in rows),
        total_chunks=sum(row["total_chunks"] for row in rows),
        changed_selections=sum(row["selection_changed"][name] for row in rows),
        complete=sum(complete(row,name) for row in rows), incomplete=sum(missing(row,name) for row in rows),
        ambiguous=sum(row["coverage"][name] == "ambiguous" for row in rows),
        unavailable=sum(row["coverage"][name] == "unavailable" for row in rows),
        newly_complete=gained, lost_complete=lost, net_complete=gained-lost,
        full_only_cases=len(full_only),
        full_only_newly_complete=sum(missing(row,"entity25") and complete(row,name) for row in full_only),
        full_only_lost_complete=sum(complete(row,"entity25") and missing(row,name) for row in full_only),
        teacher_only_cases=len(teacher_only),
        teacher_only_lost_complete=sum(complete(row,"entity25") and missing(row,name) for row in teacher_only))


def decision(stats):
    # This threshold was fixed before the initial candidate coverage sweep.
    # proper_where was designed after reviewing those development results.
    eligible = (stats["net_complete"] >= 5 and stats["full_only_newly_complete"] >= 3 and
                stats["teacher_only_lost_complete"] <= 2 and
                stats["ambiguous"] == 0 and stats["unavailable"] == 0)
    return dict(eligible_for_model_test=eligible, minimum_net_complete=5,
                minimum_full_only_newly_complete=3, maximum_teacher_only_lost_complete=2)


def collect(workspace):
    root = workspace / "query_cartridges_linked_selection_20261104"
    manifest = json.loads((root / "manifest.json").read_text())
    cases = manifest["case_ids"]
    if (len(cases) != 200 or len(set(cases)) != 200 or manifest["keep_fraction"] != .1 or
        manifest["selection_policy"] != "decision_entity_link_v1" or
        manifest["entity_reserve_fraction"] != .25 or manifest["online_api_calls"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False):
        raise ValueError("The 200-case linked development protocol changed")
    protocol = json.loads(Path(manifest["model_protocols"]["qwen"]).read_text())
    index = {row["key"]: row for row in json.loads(Path(protocol["index"]).read_text())}
    root_scores = Path(protocol["shared_scores"])
    score_files = json.loads((root_scores / "case_to_score_file.json").read_text())
    prior = {row["case"]: row for row in json.loads((root / "coverage.json").read_text())["rows"]}
    helper = workspace / "query_cartridges_20261103/audit_babilong_selected_support.py"
    spec = importlib.util.spec_from_file_location("saved_support_audit", helper)
    support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)
    panel = {row["case"]: row for row in json.loads((support.ORIGINAL / "fresh_confirmation.json").read_text())}
    units, matches = support.source_matches(json.loads(support.RESERVATION.read_text()))
    references = json.loads((workspace.parent / "chunked_ntp_query_only_20261029/evaluation_references.json").read_text())
    sys.path.insert(0, str(workspace / "monitor"))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected answer scorer")
    rows = []
    with open(protocol["source_cases"], "rb") as source:
        for key in sorted(cases, key=lambda case: index[case]["offset"]):
            entry = index[key]; source.seek(entry["offset"])
            case = json.loads(source.read(entry["bytes"]))
            saved = json.loads((root / "cases/qwen" / key / "result.json").read_text())
            scored = json.loads((root_scores / "scores" / score_files[key]).read_text())
            unit = panel[key]["sampling_unit"]
            if (case["key"] != key or case["query"].strip() != units[unit]["question"].strip() or
                scored["case_key"] != key or scored["question"] != case["query"] or
                scored["reference_answers_used"] is not False or saved["key"] != key or
                saved["selection"]["reference_answers_used"] or saved["selection"]["online_api_calls"] != 0):
                raise ValueError("Saved Decision scores or question identities changed")
            spans, records = scored["source_spans"], scored["records"]
            # These three selections see only the document, question and Decision scores.
            selections = dict(entity25=choose(case["document"],case["query"],spans,records,.1,"entity",.25),
                mix=question_overlap(case["document"],case["query"],spans,records,.1,mode="mix"),
                lexical=question_overlap(case["document"],case["query"],spans,records,.1,mode="lexical"),
                proper_where=question_overlap(case["document"],case["query"],spans,records,.1,mode="proper_where"))
            selected_old = sorted(i for group in saved["selection"]["selected_original_chunks"] for i in group)
            if (selections["entity25"] != selected_old or selections["entity25"] != prior[key]["outcomes"]["entity10"]["selected"] or
                len({len(value) for value in selections.values()}) != 1):
                raise ValueError("The saved baseline or same-chunk budget changed")
            # Annotations enter only after all candidate selections are fixed.
            coverage = {}
            for name, selected in selections.items():
                ranges = [tuple(spans[i]) for i in selected]
                coverage[name] = support.combine([support.support_coverage(case["document"], match, ranges,
                    strip_boundary_punctuation=True) for match in matches[unit]])["status"]
            if coverage["entity25"] != prior[key]["outcomes"]["entity10"]["status"]:
                raise ValueError("Existing baseline support coverage changed")
            reference = references[key]
            grade = lambda text: int(score_prediction(text, reference["answers"], "babilong", reference["task"])[0])
            rows.append(dict(task=reference["task"], length=entry["length_bucket"],
                source=key.split("-",2)[2], selected_count=len(selections["entity25"]), total_chunks=len(spans),
                whole=grade(saved["self_judge"]["native_prediction"]), teacher=grade(saved["teacher"]["prediction"]),
                selection_changed={name: selected != selections["entity25"] for name,selected in selections.items()},
                coverage=coverage))
    if len(rows) != 200 or len({row["source"] for row in rows}) != 50:
        raise ValueError("The development set is incomplete")
    subsets = defaultdict(list)
    for row in rows:
        subsets["all"].append(row);subsets["task:"+row["task"]].append(row)
        subsets["length:"+str(row["length"])].append(row)
    output = {group: {name: summarize(items,name) for name in VARIANTS} for group,items in sorted(subsets.items())}
    return dict(cases=200, source_questions=50, fraction=.1, variants=list(VARIANTS), groups=output,
        decisions={name:decision(output["all"][name]) for name in ("mix","lexical","proper_where")},
        variants_designed_after_reviewing_development_results=["proper_where"],
        support_and_answers_used_only_posthoc=True, selection_accepts_no_labels=True,
        accuracy_of_new_selections_not_measured=True, online_api_calls=0, new_model_calls=0,
        prior_other_selection_cohort_analysis_seen=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["groups"]["all"], indent=2));print("decisions",json.dumps(result["decisions"]))


if __name__ == "__main__":
    main()
