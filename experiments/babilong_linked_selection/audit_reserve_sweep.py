#!/usr/bin/env python3
"""Measure support coverage at a fixed 10% Decision-guided chunk budget.

The selector uses only the question, source text and cached Decision scores.
Annotated support facts are consulted after selection; no model is invoked.
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from masked_ttt.linked_selection import choose

RESERVES = (0., .125, .25, .5)


def summarize(rows, reserve):
    counts = Counter()
    for row in rows:
        old, now = row["outcomes"][0.], row["outcomes"][reserve]
        counts["cases"] += 1
        counts[now["status"]] += 1
        counts["new_complete"] += old["status"] == "incomplete" and now["status"] == "complete"
        counts["lost_complete"] += old["status"] == "complete" and now["status"] == "incomplete"
        counts["changed_selection"] += old["selected"] != now["selected"]
        counts["selected_chunks"] += len(now["selected"])
        counts["total_chunks"] += row["total_chunks"]
    return {key: int(counts[key]) for key in ("cases", "complete", "incomplete", "ambiguous", "unavailable",
            "new_complete", "lost_complete", "changed_selection", "selected_chunks", "total_chunks")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "panel", "reservation", "archive", "support-script", "existing-coverage", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    chosen = set(manifest["case_ids"])
    if len(chosen) != 200 or len(manifest["case_ids"]) != 200 or manifest["keep_fraction"] != .1:
        raise ValueError("The fixed 200-case, 10% development split is required")
    protocol = json.loads(Path(manifest["model_protocols"]["qwen"]).read_text())
    index = {row["key"]: row for row in json.loads(Path(protocol["index"]).read_text())}
    panel = {row["case"]: row for row in json.loads(args.panel.read_text())}
    scores_root = Path(protocol["shared_scores"])
    files = json.loads((scores_root / "case_to_score_file.json").read_text())
    previous = {row["case"]: row for row in json.loads(args.existing_coverage.read_text())["rows"]}
    if set(previous) != chosen or not chosen.issubset(index.keys() & panel.keys() & files.keys()):
        raise ValueError("Development inputs and prior coverage do not match")
    spec = importlib.util.spec_from_file_location("support_audit", args.support_script)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    support.ARCHIVE = args.archive
    units, matches = support.source_matches(json.loads(args.reservation.read_text()))
    rows = []
    with open(protocol["source_cases"], "rb") as source:
        for key in sorted(chosen, key=lambda case: index[case]["offset"]):
            entry = index[key]
            source.seek(entry["offset"])
            case = json.loads(source.read(entry["bytes"]))
            unit = panel[key]["sampling_unit"]
            if case["key"] != key or case["query"].strip() != units[unit]["question"].strip():
                raise ValueError("The question differs from its source annotation")
            data = json.loads((scores_root / "scores" / files[key]).read_text())
            if (data.get("case_key") != key or data.get("question") != case["query"] or
                data.get("reference_answers_used") is not False):
                raise ValueError("The cached Decision identity changed")
            spans, records = data["source_spans"], data["records"]
            selections = {r: choose(case["document"], case["query"], spans, records, .1, "entity", r)
                          for r in RESERVES}
            if any(len(ids) != max(1, math.ceil(len(records) * .1)) for ids in selections.values()):
                raise ValueError("The chunk budget changed across selection variants")
            # Only now use annotated support; it never enters choose().
            outcomes = {}
            for reserve, selected in selections.items():
                ranges = [tuple(spans[i]) for i in selected]
                coverage = support.combine([support.support_coverage(case["document"], match, ranges,
                    strip_boundary_punctuation=True) for match in matches[unit]])
                outcomes[reserve] = dict(status=coverage["status"], selected=selected)
            for reserve, name in ((0., "score10"), (.25, "entity10")):
                old = previous[key]["outcomes"][name]
                if outcomes[reserve]["selected"] != old["selected"] or outcomes[reserve]["status"] != old["status"]:
                    raise ValueError("The existing paired selection or support coverage changed")
            rows.append(dict(case=key, task=entry["task"], length=entry["length_bucket"],
                             total_chunks=len(records), outcomes=outcomes))
    if len(rows) != 200 or len({row["case"].split("-", 2)[2] for row in rows}) != 50:
        raise ValueError("Expected 200 cases from 50 source questions")
    grouped = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        grouped["task:" + row["task"]].append(row)
        grouped["length:" + str(row["length"])].append(row)
    output = dict(scope="fixed 200-case development split; 50 source questions at four lengths",
        keep_fraction=.1, reserve_fractions=list(RESERVES), online_api_calls=0, new_model_calls=0,
        reference_answers_used=False, support_annotations_used_only_after_selection=True,
        prior_score_and_025_linked_cases_validated=200,
        groups={name: {str(r): summarize(items, r) for r in RESERVES} for name, items in sorted(grouped.items())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["groups"]["all"], indent=2))


if __name__ == "__main__":
    main()
