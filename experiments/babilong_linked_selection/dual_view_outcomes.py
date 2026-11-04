"""Audit same-question whole/selected/cartridge complementarity in a saved report."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(rows):
    names = ("whole", "zero", "teacher", "cartridge")
    if any(type(row[key]) is not int or row[key] not in (0, 1) for row in rows for key in names):
        raise ValueError("Expected binary post-generation scores")
    n = len(rows)
    whole_only = [row for row in rows if row["whole"] and not row["zero"]]
    selected_only = [row for row in rows if row["zero"] and not row["whole"]]
    def pair(new, old):
        gains = sum(row[new] and not row[old] for row in rows)
        losses = sum(row[old] and not row[new] for row in rows)
        return dict(new_correct=gains, new_wrong=losses,
                    delta_pp=100 * (gains-losses)/n if n else None)
    return dict(n=n, source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
        accuracy={key: 100 * sum(row[key] for row in rows)/n if n else None for key in names},
        whole_only=dict(n=len(whole_only), cartridge_correct=sum(row["cartridge"] for row in whole_only)),
        selected_only=dict(n=len(selected_only), cartridge_correct=sum(row["cartridge"] for row in selected_only)),
        versus_zero=pair("cartridge", "zero"), versus_teacher=pair("cartridge", "teacher"))


def collect(report):
    if (report["metric"] != "answer_focused_lexical_v2" or not report["development_only"] or
        report["expected_per_model"] != 200 or report["selection_policy"] != "decision_entity_link_v1" or
        not report["cached_scores"] or report["online_api_calls"] != 0 or
        not report["evaluation_answers_used_only_posthoc"] or report["errors"] or report["invalid"]):
        raise ValueError("Expected the clean linked-selection development report")
    models = {}
    for model in ("qwen", "gpt"):
        rows = report["models"][model]["rows"]
        if len(rows) != 200 or len({row["case"] for row in rows}) != 200:
            raise ValueError("Each model must have exactly 200 scored cases")
        groups = defaultdict(list)
        for row in rows:
            groups[row["task"]].append(row)
        models[model] = dict(overall=summarize(rows), groups={task: summarize(items)
                            for task, items in sorted(groups.items())})
    return dict(metric=report["metric"], source_snapshot=report["updated"], selected_fraction=.1,
                references_used_only_posthoc=True, new_model_calls=0, online_api_calls=0, models=models)


def render(result):
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    lines = ["# Full and selected views: paired development outcomes", "",
        "| Model / type | n | Whole | Selected zero | Teacher | Cartridge | Whole-only recovered | Selected-only retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, item in result["models"].items():
        for name, group in [(model, item["overall"]), (model + " QA3", item["groups"]["qa3"])]:
            acc = group["accuracy"]; whole = group["whole_only"]; selected = group["selected_only"]
            lines.append(f"| {name} | {group['n']} | {fmt(acc['whole'])} | {fmt(acc['zero'])} | "
                f"{fmt(acc['teacher'])} | {fmt(acc['cartridge'])} | "
                f"{whole['cartridge_correct']} / {whole['n']} | "
                f"{selected['cartridge_correct']} / {selected['n']} |")
    lines += ["", "Whole-only and selected-only are determined by post-generation labels on the same case. "
              "A recovery means the final Cartridge answer was correct in a whole-only case; retention is defined "
              "analogously on selected-only cases. Teacher has the original generic formatting requirement, "
              "whereas whole, selected zero, and Cartridge use the same original question format.",
              "Each 200-case model subset contains 50 source questions at four context lengths; "
              "these are development results and do not establish why a case changed.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = collect(json.loads(args.report.read_text()))
    for suffix, text in ((".json", json.dumps(result, indent=2) + "\n"), (".md", render(result))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
    print(render(result))


if __name__ == "__main__":
    main()
