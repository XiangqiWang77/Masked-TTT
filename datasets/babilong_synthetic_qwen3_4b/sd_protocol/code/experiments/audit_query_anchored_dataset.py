"""Audit query-anchored synthetic data without making model calls.

The audit compares completed cases from a new answer-guided run with the
same cases in a legacy answer-guided run and a fixed, count-matched vanilla
run.  It reads evaluation-only BABILong support positions for exposure
diagnostics, but never reads benchmark answers and never changes generation.

The resulting gate is deliberately a *dataset* gate.  Passing it is not a
claim that Cartridge accuracy improves; that claim requires the paired 2x2
SD experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median

from contextualized.data import read_indexed_case
from experiments.analyze_synthetic_comparison import TOKENS, normalized, query_terms
from experiments.report_answer_guided_remaining import (
    arm_metrics,
    checked_facts,
    contexts_from_original,
)


SCHEMA = "query-anchored-dataset-audit-v1"
DEFAULT_NEW = "/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/answer_guided_query_anchored_v3_20261109"
DEFAULT_OLD = "/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/answer_guided_api_remaining_20261108"
DEFAULT_VANILLA = "/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/vanilla_syns_matched_1000_20261108"


def read(path: str | Path):
    return json.loads(Path(path).read_text())


def atomic_write(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def avg(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def assistant_text(example) -> str:
    if isinstance(example.get("answer"), str):
        return example["answer"]
    for message in example.get("messages", []):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def question_text(example) -> str:
    if isinstance(example.get("question"), str):
        return example["question"]
    for message in example.get("messages", []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def guided_examples(marker):
    examples, bodies = [], []
    for split in ("train", "validation"):
        body = read(marker["paths"][split])
        if body.get("case") != marker["case"] or body.get("actual_count") != len(body.get("examples", [])):
            raise ValueError(f"inconsistent guided {split} split for {marker['case']}")
        examples.extend(body["examples"])
        bodies.append(body)
    if len(examples) != marker["counts"]["answer_guided"]:
        raise ValueError(f"guided count mismatch for {marker['case']}")
    return examples, bodies


def vanilla_examples(marker):
    examples, bodies = [], []
    for split in ("original_train", "original_validation"):
        body = read(marker["paths"][split])
        if body.get("case") != marker["case"] or body.get("actual_count") != len(body.get("conversations", [])):
            raise ValueError(f"inconsistent vanilla {split} split for {marker['case']}")
        examples.extend(body["conversations"])
        bodies.append(body)
    return examples, bodies


def recursive_true_benchmark_flag(value) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if "benchmark_answer" in key.casefold() and item is True:
                return True
            if recursive_true_benchmark_flag(item):
                return True
    elif isinstance(value, list):
        return any(recursive_true_benchmark_flag(item) for item in value)
    return False


def evidence_is_anchored(example):
    evidence = example.get("evidence")
    if not isinstance(evidence, dict):
        return None
    chunks = evidence.get("source_chunks")
    if not isinstance(chunks, list) or not chunks:
        return False
    return any(
        chunk.get("decision_api_accepted") is True
        or isinstance(chunk.get("lexical_score"), (int, float)) and chunk["lexical_score"] > 0
        for chunk in chunks
    )


def local_metrics(examples, query, *, guided):
    answers = [normalized(assistant_text(example)) for example in examples]
    answers = [answer for answer in answers if answer]
    answer_counts = Counter(answers)
    questions = [question_text(example) for example in examples]
    terms = set(query_terms(query))
    anchors = [evidence_is_anchored(example) for example in examples] if guided else []
    chunk_counts = [len(example.get("evidence", {}).get("source_chunks", [])) for example in examples] if guided else []
    return {
        "questions": len(examples),
        "distinct_answers": len(answer_counts),
        "distinct_answer_fraction": len(answer_counts) / len(answers) if answers else None,
        "dominant_answer_fraction": max(answer_counts.values()) / len(answers) if answers else None,
        "query_term_hit_fraction": (
            sum(bool(terms & set(TOKENS.findall(question.casefold()))) for question in questions) / len(questions)
            if terms and questions else None
        ),
        "exact_question_duplicate_fraction": (
            (len(questions) - len({normalized(question) for question in questions})) / len(questions)
            if questions else None
        ),
        "anchored_evidence_fraction": avg([float(value) for value in anchors if value is not None]),
        "unanchored_evidence_examples": sum(value is False for value in anchors),
        "multi_chunk_evidence_fraction": avg([float(count > 1) for count in chunk_counts]),
    }


def aggregate(rows, arm):
    values = [row[arm] for row in rows]
    distinct = [value["local"]["distinct_answers"] for value in values]
    support = [value["support"] for value in values]
    return {
        "cases": len(values),
        "questions": sum(value["local"]["questions"] for value in values),
        "questions_per_case_mean": avg([value["local"]["questions"] for value in values]),
        "distinct_answers_per_case_mean": avg(distinct),
        "distinct_answers_per_case_median": median(distinct) if distinct else None,
        "distinct_answer_fraction_mean": avg([value["local"]["distinct_answer_fraction"] for value in values]),
        "dominant_answer_fraction_mean": avg([value["local"]["dominant_answer_fraction"] for value in values]),
        "query_term_hit_fraction_mean": avg([value["local"]["query_term_hit_fraction"] for value in values]),
        "exact_question_duplicate_fraction_mean": avg(
            [value["local"]["exact_question_duplicate_fraction"] for value in values]
        ),
        "anchored_evidence_fraction_mean": avg([value["local"]["anchored_evidence_fraction"] for value in values]),
        "unanchored_evidence_examples": sum(value["local"]["unanchored_evidence_examples"] for value in values),
        "multi_chunk_evidence_fraction_mean": avg(
            [value["local"]["multi_chunk_evidence_fraction"] for value in values]
        ),
        "support_recall_lower_bound_mean": avg(
            [value.get("context_union_support_recall_min") for value in support]
        ),
        "all_support_context_fraction_lower_bound_mean": avg(
            [value.get("complete_context_fraction_min") for value in support]
        ),
        "no_support_context_fraction_mean": avg([value.get("no_support_context_fraction") for value in support]),
    }


def load_annotations(path):
    result = {}
    if not Path(path).is_file():
        return result
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["case"] in result:
                raise ValueError(f"duplicate support annotation for {row['case']}")
            result[row["case"]] = row
    return result


def build_report(new_root, old_root, vanilla_root, *, min_cases, max_cases=None):
    new_root, old_root, vanilla_root = map(Path, (new_root, old_root, vanilla_root))
    config = read(new_root / "config.json")
    plan = read(config["plan"])
    planned = {row["key"]: row for row in plan["cases"]}
    source_index = {row["key"]: row for row in read(plan["source_index"])}
    annotation_path = Path(config.get("report_support_annotations") or Path(config["plan"]).parent / "evaluation/source_annotations.jsonl")
    annotations = load_annotations(annotation_path)
    rows, issues = [], []
    for path in sorted((new_root / "dataset_cases").glob("*.json")):
        if max_cases is not None and len(rows) >= max_cases:
            break
        try:
            new_marker = read(path)
            if new_marker.get("complete") is not True:
                continue
            key = new_marker["case"]
            if key not in planned:
                raise ValueError(f"unplanned case {key}")
            old_marker = read(old_root / "dataset_cases" / f"{key}.json")
            vanilla_marker = read(vanilla_root / "dataset_cases" / f"{key}.json")
            if old_marker.get("complete") is not True or vanilla_marker.get("complete") is not True:
                raise ValueError(f"comparison arm incomplete for {key}")
            if len({new_marker.get("query"), old_marker.get("query"), vanilla_marker.get("query")}) != 1:
                raise ValueError(f"query identity mismatch for {key}")
            query = new_marker["query"]
            new_examples, new_bodies = guided_examples(new_marker)
            old_examples, old_bodies = guided_examples(old_marker)
            vanilla, vanilla_bodies = vanilla_examples(vanilla_marker)
            matched_new, matched_old, matched_vanilla = [], [], []
            new_has_matched_budget = True
            for new_body, old_body, vanilla_body in zip(new_bodies, old_bodies, vanilla_bodies):
                old_split = old_body["examples"]
                vanilla_split = vanilla_body["conversations"]
                if len(old_split) != len(vanilla_split):
                    raise ValueError(f"old-guided/vanilla split budget mismatch for {key}")
                new_split = new_body["examples"]
                new_has_matched_budget &= len(new_split) >= len(old_split)
                common = min(len(new_split), len(old_split))
                matched_new.extend(new_split[:common])
                matched_old.extend(old_split[:common])
                matched_vanilla.extend(vanilla_split[:common])
            case = {**read_indexed_case(plan, source_index, key),
                    "task": planned[key]["task"], "length": planned[key]["length"]}
            if case["query"] != query:
                raise ValueError(f"source query mismatch for {key}")
            facts = checked_facts(annotations.get(key), case)
            contexts = contexts_from_original(read(vanilla_marker["paths"]["source"]), case["document"])
            arm_examples = {
                "new_guided": matched_new,
                "old_guided": matched_old,
                "vanilla": matched_vanilla,
            }
            row = {
                "case": key,
                "task": planned[key]["task"],
                "length": planned[key]["length"],
                "benchmark_answer_flag_true": recursive_true_benchmark_flag(
                    [new_marker, *new_bodies]
                ),
                "new_has_matched_budget": new_has_matched_budget,
                "new_has_nonempty_train_and_validation": all(
                    len(body["examples"]) > 0 for body in new_bodies
                ),
                "new_full_local": local_metrics(new_examples, query, guided=True),
                "new_full_support": arm_metrics(new_examples, case, facts, contexts),
            }
            for arm, examples in arm_examples.items():
                row[arm] = {
                    "local": local_metrics(examples, query, guided=arm != "vanilla"),
                    "support": arm_metrics(examples, case, facts, contexts),
                }
            rows.append(row)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            issues.append({"path": str(path), "error": str(error)})

    summaries = {arm: aggregate(rows, arm) for arm in ("vanilla", "old_guided", "new_guided")}
    cross_rows = [row for row in rows if row["new_guided"]["support"].get("has_complete_context_min") is not None
                  and annotations.get(row["case"], {}).get("support_requires_multiple_original_chunks") is True]
    # Older annotation files do not always materialize the convenience flag.
    if not cross_rows:
        cross_rows = [row for row in rows if len([
            fact for fact in annotations.get(row["case"], {}).get("facts", []) if fact.get("is_support")
        ]) > 1]
    cross_summaries = {arm: aggregate(cross_rows, arm) for arm in ("vanilla", "old_guided", "new_guided")}

    new = summaries["new_guided"]
    old = summaries["old_guided"]
    vanilla = summaries["vanilla"]
    checks = {
        "enough_completed_cases": len(rows) >= min_cases,
        "no_benchmark_answer_flags": not any(row["benchmark_answer_flag_true"] for row in rows),
        "all_new_exercises_query_anchored": sum(
            row["new_full_local"]["unanchored_evidence_examples"] for row in rows
        ) == 0,
        "new_dataset_has_nonempty_train_and_validation": all(
            row["new_has_nonempty_train_and_validation"] for row in rows
        ) if rows else False,
        "distinct_answers_improve_over_old_guided": (
            new["distinct_answers_per_case_mean"] is not None
            and old["distinct_answers_per_case_mean"] is not None
            and new["distinct_answers_per_case_mean"] > old["distinct_answers_per_case_mean"]
        ),
        "query_focus_improves_over_vanilla": (
            new["query_term_hit_fraction_mean"] is not None
            and vanilla["query_term_hit_fraction_mean"] is not None
            and new["query_term_hit_fraction_mean"] > vanilla["query_term_hit_fraction_mean"]
        ),
        "support_recall_does_not_regress_from_old_guided": (
            new["support_recall_lower_bound_mean"] is not None
            and old["support_recall_lower_bound_mean"] is not None
            and new["support_recall_lower_bound_mean"] + 1e-12 >= old["support_recall_lower_bound_mean"]
        ),
        "cross_fact_all_support_exposure_improves_over_vanilla": (
            len(cross_rows) >= min(20, min_cases)
            and cross_summaries["new_guided"]["all_support_context_fraction_lower_bound_mean"] is not None
            and cross_summaries["vanilla"]["all_support_context_fraction_lower_bound_mean"] is not None
            and cross_summaries["new_guided"]["all_support_context_fraction_lower_bound_mean"]
            > cross_summaries["vanilla"]["all_support_context_fraction_lower_bound_mean"]
        ),
    }
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {"new_guided": str(new_root), "old_guided": str(old_root), "vanilla": str(vanilla_root)},
        "completed_cases": len(rows),
        "planned_cases": len(planned),
        "minimum_cases_for_gate": min_cases,
        "maximum_cases_requested": max_cases,
        "support_annotations": str(annotation_path) if annotation_path.is_file() else None,
        "benchmark_answers_read": False,
        "summaries": summaries,
        "new_full_dataset_summary": {
            "questions": sum(row["new_full_local"]["questions"] for row in rows),
            "distinct_answers_per_case_mean": avg(
                [row["new_full_local"]["distinct_answers"] for row in rows]
            ),
            "dominant_answer_fraction_mean": avg(
                [row["new_full_local"]["dominant_answer_fraction"] for row in rows]
            ),
            "unanchored_evidence_examples": sum(
                row["new_full_local"]["unanchored_evidence_examples"] for row in rows
            ),
            "support_recall_lower_bound_mean": avg([
                row["new_full_support"].get("context_union_support_recall_min") for row in rows
            ]),
            "all_support_context_fraction_lower_bound_mean": avg([
                row["new_full_support"].get("complete_context_fraction_min") for row in rows
            ]),
            "multi_chunk_evidence_fraction_mean": avg([
                row["new_full_support"].get("multisource_context_fraction") for row in rows
            ]),
            "cases_meeting_old_per_split_question_budget": sum(
                row["new_has_matched_budget"] for row in rows
            ),
            "comparison_note": (
                "paired arm metrics use the same deterministic train/validation prefix in all arms; "
                "the complete v3 dataset is reported separately"
            ),
        },
        "cross_fact_subset": {"cases": len(cross_rows), "summaries": cross_summaries},
        "checks": checks,
        "dataset_gate_passed": all(checks.values()),
        "performance_claim_established": False,
        "performance_claim_requires": "paired 2x2 native-KV SD development experiment",
        "issues": issues,
        "cases": rows,
    }


def markdown(report):
    def pct(value):
        return "—" if value is None else f"{100 * value:.2f}%"

    def number(value):
        return "—" if value is None else f"{value:.2f}"

    lines = [
        "# Query-anchored synthetic-data audit",
        "",
        f"Completed cases: {report['completed_cases']}/{report['planned_cases']}. Dataset gate: **{'PASS' if report['dataset_gate_passed'] else 'NOT YET PASSED'}**.",
        "",
        "| Arm | Questions | Distinct answers/case | Dominant answer share | Query-term focus | Support recall (LB) | All-support context (LB) | Unanchored examples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = (("vanilla", "Vanilla"), ("old_guided", "Old guided v2"), ("new_guided", "New guided v3"))
    for key, label in labels:
        row = report["summaries"][key]
        lines.append(
            f"| {label} | {row['questions']} | {number(row['distinct_answers_per_case_mean'])} | "
            f"{pct(row['dominant_answer_fraction_mean'])} | {pct(row['query_term_hit_fraction_mean'])} | "
            f"{pct(row['support_recall_lower_bound_mean'])} | "
            f"{pct(row['all_support_context_fraction_lower_bound_mean'])} | "
            f"{row['unanchored_evidence_examples']} |"
        )
    lines += ["", "Checks:"]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines += [
        "",
        "This gate audits dataset focus, source exposure, answer-target diversity, and provenance. "
        "It does not establish semantic correctness or downstream accuracy. The two method claims "
        "remain closed until the paired 2x2 native-KV SD experiment is positive.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", default=DEFAULT_NEW)
    parser.add_argument("--old-root", default=DEFAULT_OLD)
    parser.add_argument("--vanilla-root", default=DEFAULT_VANILLA)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--max-cases", type=int, help="deterministic prefix for an early snapshot")
    parser.add_argument("--output", help="default: <new-root>/quality_gate")
    args = parser.parse_args()
    report = build_report(args.new_root, args.old_root, args.vanilla_root,
                          min_cases=args.min_cases, max_cases=args.max_cases)
    output = Path(args.output) if args.output else Path(args.new_root) / "quality_gate"
    atomic_write(output / "report.json", report)
    summary = output / "summary.md"
    temporary = summary.with_name(summary.name + f".{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(markdown(report))
    temporary.replace(summary)
    print(json.dumps({
        "completed_cases": report["completed_cases"],
        "dataset_gate_passed": report["dataset_gate_passed"],
        "checks": report["checks"],
        "report": str(output / "report.json"),
    }))


if __name__ == "__main__":
    main()
