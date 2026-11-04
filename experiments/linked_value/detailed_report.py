#!/usr/bin/env python3
"""Report GPT linked-value quality by task, context length and source question.

Use only the fixed post-generation binary scores, never source text or answers.
The output contains aggregates and does not expose case identifiers.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ARMS = ("whole", "zero", "teacher", "cartridge")


def summarize(rows):
    if any(type(row.get(name)) is not int or row[name] not in (0, 1) for row in rows for name in ARMS):
        raise ValueError("Expected binary paired answer scores")
    n = len(rows)
    accuracy = {name: 100 * sum(row[name] for row in rows) / n if n else None for name in ARMS}
    pairs = {}
    for base in ("whole", "zero", "teacher"):
        good = sum(not row[base] and row["cartridge"] for row in rows)
        bad = sum(row[base] and not row["cartridge"] for row in rows)
        groups = defaultdict(list)
        for row in rows:
            groups[row["case"].split("-", 2)[2]].append(row)
        outcomes = Counter()
        for group in groups.values():
            source_good = any(not row[base] and row["cartridge"] for row in group)
            source_bad = any(row[base] and not row["cartridge"] for row in group)
            label = ("mixed" if source_good and source_bad else "improved_only" if source_good else
                     "regressed_only" if source_bad else "unchanged")
            outcomes[label] += 1
        pairs[base] = dict(new_correct=good, new_wrong=bad,
            delta_pp=100 * (good-bad) / n if n else None,
            observed_sources=len(groups), source_outcomes={key: outcomes[key]
                for key in ("improved_only", "regressed_only", "mixed", "unchanged")})
    return dict(n=n, observed_sources=len({row["case"].split("-", 2)[2] for row in rows}),
                accuracy=accuracy, versus=pairs)


def validate_rows(rows, allowed, saved):
    keys = [row["case"] for row in rows]
    if len(set(keys)) != len(keys) or not set(keys).issubset(allowed):
        raise ValueError("A score is duplicated or outside the fixed case set")
    for row in rows:
        parts = row["case"].split("-", 2)
        if (len(parts) != 3 or parts[0] != "babilong" or parts[1] not in ("8", "16", "32", "64") or
            row["length"] != int(parts[1]) * 1024 or row["task"] != parts[2].split("-", 1)[0]):
            raise ValueError("A saved case has an inconsistent task or length")
    computed = summarize(rows)
    if saved["n"] != computed["n"]:
        raise ValueError("The saved score count differs from its rows")
    for arm in ARMS:
        a, b = saved["accuracy"][arm], computed["accuracy"][arm]
        if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, abs_tol=1e-9)):
            raise ValueError("The saved aggregate accuracy differs from its rows")
    for base in ("whole", "zero", "teacher"):
        old, new = saved["versus_" + base], computed["versus"][base]
        if old["new_correct"] != new["new_correct"] or old["new_wrong"] != new["new_wrong"]:
            raise ValueError("The saved paired changes differ from their rows")
    return computed


def sections(rows):
    by_task, by_length = defaultdict(list), defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
        by_length[row["length"]].append(row)
    return dict(overall=summarize(rows),
        by_task={key: summarize(value) for key, value in sorted(by_task.items(),
                 key=lambda item: (int(item[0][2:]) if item[0].startswith("qa") and item[0][2:].isdigit() else 999,
                                   item[0]))},
        by_length={str(key): summarize(value) for key, value in sorted(by_length.items())})


def collect(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "latest.json").read_text())
    if (manifest["mode"] != "holdout" or manifest["models"] != ["gpt"] or manifest["case_count"] != 800 or
        manifest["case_ids"] != manifest["holdout_case_ids"] or manifest["fixed_value_learning_rate"] != .001 or
        manifest["selection_policy"] != "decision_entity_link_v1" or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        report["mode"] != "holdout" or report["model"] != "gpt" or report["expected"] != 800 or
        report["metric"] != "answer_focused_lexical_v2" or report["online_api_calls"] != 0):
        raise ValueError("The fixed GPT holdout protocol changed")
    held = report["rows"]
    validate_rows(held, set(manifest["holdout_case_ids"]), report["overall"])
    development = json.loads((root / "development_snapshot.json").read_text())
    if (development["mode"] != "development" or development["model"] != "gpt" or
        development["expected"] != 200 or development["metric"] != report["metric"] or
        development["online_api_calls"] != 0 or development["errors"] or development["invalid"]):
        raise ValueError("The saved development report changed")
    dev = development["rows"]
    validate_rows(dev, set(manifest["development_case_ids"]), development["overall"])
    dev_keys, held_keys = {row["case"] for row in dev}, {row["case"] for row in held}
    if (len(dev) != 200 or dev_keys != set(manifest["development_case_ids"]) or dev_keys & held_keys or
        {key.split("-", 2)[2] for key in dev_keys} & {key.split("-", 2)[2] for key in held_keys}):
        raise ValueError("Development and holdout overlap or are incomplete")
    complete = len(held) == 800 and held_keys == set(manifest["holdout_case_ids"]) and not report["errors"] and not report["invalid"]
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), model="GPT-OSS-20B",
        metric=report["metric"], online_api_calls=0, new_model_calls=0, reference_answers_used_only_posthoc=True,
        teacher_and_original_cartridge_formats_differ=True, development_used_for_selection=True,
        source_disjoint_holdout=True, lengths_share_source_questions=True,
        expected_holdout=800, holdout_complete=complete,
        invalid_count=len(report["invalid"]), error_count=len(report["errors"]),
        development=sections(dev), holdout=sections(held))
    if complete:
        output["combined_1000"] = sections(dev + held)
        viewed = set(manifest["qualitative_case_sources"])
        unviewed = [row for row in held if row["case"].split("-", 2)[2] not in viewed]
        if len(unviewed) != 792:
            raise ValueError("Unexpected previously inspected source exclusion")
        output["holdout_excluding_previously_viewed_sources"] = sections(unviewed)
    return output


def render(data):
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"
    def line(label, stats):
        a, t, z = stats["accuracy"], stats["versus"]["teacher"], stats["versus"]["zero"]
        return (f"| {label} | {stats['n']} | {fmt(a['whole'])} | {fmt(a['zero'])} | {fmt(a['teacher'])} | "
                f"{fmt(a['cartridge'])} | {fmt(t['delta_pp'])} | {t['new_correct']} / {t['new_wrong']} | "
                f"{z['new_correct']} / {z['new_wrong']} |")
    lines = ["# GPT-OSS linked-value paired answer quality", "",
        f"Metric: {data['metric']}; holdout: {data['holdout']['overall']['n']} / {data['expected_holdout']}.", "",
        "| Scope / group | n | Whole | Selected zero | ICL Teacher | Cartridge | Δ vs Teacher | New right / wrong vs Teacher | New right / wrong vs zero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("development", "holdout", "combined_1000", "holdout_excluding_previously_viewed_sources"):
        if name not in data:
            continue
        part = data[name]
        lines.append(line(name, part["overall"]))
        for task, stats in part["by_task"].items():
            lines.append(line(name + " " + task.upper(), stats))
        for length, stats in part["by_length"].items():
            lines.append(line(name + " " + str(int(length)//1024) + "K", stats))
    lines += ["", "| Scope | Compared with | Observed source questions | Improved only | Regressed only | Mixed | Unchanged |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("development", "holdout", "combined_1000", "holdout_excluding_previously_viewed_sources"):
        if name not in data:
            continue
        for base in ("teacher", "zero"):
            pair = data[name]["overall"]["versus"][base]
            out = pair["source_outcomes"]
            lines.append(f"| {name} | {base} | {pair['observed_sources']} | {out['improved_only']} | "
                         f"{out['regressed_only']} | {out['mixed']} | {out['unchanged']} |")
    lines += ["", "Teacher includes a generic answer-format instruction; zero and cartridge use the same "
        "original-question format. Scores are the same post-generation loose lexical metric for every arm.",
        "Each source question occurs at up to four lengths. The JSON also groups paired gains and regressions "
        "by observed source question; partially completed holdout sources may have fewer than four lengths.",
        "The development set selected the configuration; other variants and two holdout sources were previously viewed.", ""]
    if data["invalid_count"] or data["error_count"]:
        lines += [f"Incomplete results: {data['invalid_count']} invalid cases and {data['error_count']} errors.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.job_root)
    for suffix, text in ((".json", json.dumps(data, indent=2) + "\n"), (".md", render(data))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


if __name__ == "__main__":
    main()
