"""Post-generation QA3 Teacher comparison on the frozen development cases."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys


MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def validate(case, old, new):
    if old["key"] != case or new["key"] != case or old["task"] != "qa3" or new["task"] != "qa3":
        raise ValueError("Unexpected case or question type")
    if old["model"] != MODEL or new["model"] != MODEL:
        raise ValueError("Unexpected model")
    for result in (old, new):
        s = result["selection"]
        if (s["source_case_key"] != case or s["reference_answers_used"] or s["online_api_calls"] != 0 or
            not s["complete_original_document"] or
            len(s["selected_chunks"]) != max(1, math.ceil(.1 * s["total_chunks"])) or
            not result["teacher"]["response_instruction_applied"]):
            raise ValueError("Source, 10% budget, or teacher format differs")
    if (old["selection"]["total_chunks"] != new["selection"]["total_chunks"] or
        old["selection"]["original_prompt_tokens"] != new["selection"]["original_prompt_tokens"] or
        new["selection"]["selection_policy"] != "decision_entity_link_v1" or
        new["selection"]["entity_reserve_fraction"] != .25):
        raise ValueError("The selection strategies do not share the same budget")


def audit(case_ids, old_root, new_root, refs, score):
    cases = [case for case in case_ids if "-qa3-" in case]
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate QA3 case")
    rows, missing = [], []
    for case in cases:
        old_path = Path(old_root) / "cases" / case / "weight_0.00.json"
        new_path = Path(new_root) / "cases/qwen" / case / "result.json"
        if not old_path.is_file() or not new_path.is_file():
            missing.append(case)
            continue
        old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
        validate(case, old, new)
        ref = refs[case]
        if ref["task"] != "qa3":
            raise ValueError("Evaluation question type differs")
        grade = lambda text: int(score(text, ref["answers"], "babilong", "qa3")[0])
        rows.append({"case": case, "source": case.split("-", 2)[2],
                     "score_only_teacher": grade(old["teacher"]["prediction"]),
                     "linked_teacher": grade(new["teacher"]["prediction"])})
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    def summarize(items):
        return {"n": len(items), "score_only_correct": sum(x["score_only_teacher"] for x in items),
                "linked_correct": sum(x["linked_teacher"] for x in items),
                "new_correct": sum(x["linked_teacher"] and not x["score_only_teacher"] for x in items),
                "new_wrong": sum(x["score_only_teacher"] and not x["linked_teacher"] for x in items)}
    return {"model": MODEL, "metric": "answer_focused_lexical_v2", "expected": len(cases),
            "overall": summarize(rows), "source_questions": len(groups),
            "per_source": {key: summarize(value) for key, value in sorted(groups.items())},
            "rows": rows, "missing": missing, "reference_answers_used_only_posthoc": True,
            "new_model_calls": 0, "online_api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "old-root", "new-root", "references", "scorer-dir", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected shared metric")
    result = audit(json.loads(args.manifest.read_text())["case_ids"], args.old_root, args.new_root,
                   json.loads(args.references.read_text()), score_prediction)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
