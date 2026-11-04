#!/usr/bin/env python3
"""Paired Qwen quality report for question-guided selection and original SD."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.report import validate_candidate


def summarize(rows):
    arms = ("whole", "zero", "teacher", "cartridge", "old_whole", "old_zero", "old_teacher", "old_cartridge")
    if any(any(type(row.get(key)) is not int or row[key] not in (0,1) for key in arms) for row in rows):
        raise ValueError("Expected binary answers for each paired method")
    n = len(rows)
    def change(new, old):
        gained = sum(row[new] and not row[old] for row in rows)
        lost = sum(row[old] and not row[new] for row in rows)
        return dict(new_correct=gained, new_wrong=lost, delta_pp=100*(gained-lost)/n if n else None)
    return dict(n=n, accuracy={name:100*sum(row[name] for row in rows)/n if n else None for name in arms},
        cartridge_vs_teacher=change("cartridge", "teacher"), cartridge_vs_zero=change("cartridge", "zero"),
        cartridge_vs_old=change("cartridge", "old_cartridge"), teacher_vs_old=change("teacher", "old_teacher"),
        changed_selection=sum(row["selection_changed"] for row in rows),
        same_whole_output=sum(row["same_whole_output"] for row in rows),
        same_teacher_tokens=sum(row["same_teacher_tokens"] for row in rows),
        same_zero_tokens=sum(row["same_zero_tokens"] for row in rows),
        same_cartridge_tokens=sum(row["same_cartridge_tokens"] for row in rows))


def validate_old(old, new, case):
    x,y = old["selection"], new["selection"]
    a,b = old["training"], new["training"]
    pair = old["standalone_cartridge"]["paired_prompt_control"]
    fields = ("objective", "optimizer", "teacher_cap", "teacher_target_scope", "student_rollout_limit",
              "adaptive_max_answer_steps", "adaptive_answer_KL_stop")
    if (old["key"] != case or new["key"] != case or old["model"] != new["model"] or
        old["task"] != new["task"] or x["source_case_key"] != case or y["source_case_key"] != case or
        x["selection_policy"] != "decision_entity_link_v1" or x["entity_reserve_fraction"] != .25 or
        y["selection_policy"] != "decision_proper_where_v1" or y["question_overlap_policy"] != "proper_where" or
        not x["complete_original_document"] or not y["complete_original_document"] or
        x["total_chunks"] != y["total_chunks"] or x["original_prompt_tokens"] != y["original_prompt_tokens"] or
        x["reference_answers_used"] or y["reference_answers_used"] or
        x["online_api_calls"] != 0 or y["online_api_calls"] != 0 or
        any(a.get(key) != b.get(key) for key in fields) or
        a["student_rollout_limit"] != 8 or a["adaptive_max_answer_steps"] != 8 or
        a["adaptive_answer_KL_stop"] != .5 or a["selected_transfer_weight"] != 0 or
        b["selected_transfer_weight"] != 0 or a.get("value_basis_source", "document") != "document" or
        b["value_basis_source"] != "document" or not pair["same_selected_kv"] or
        not pair["same_generation_limit"] or pair["reference_answers_used"] or
        pair["teacher_answer_in_student_prompt"] or not old["teacher"]["response_instruction_applied"]):
        raise ValueError("The paired baseline does not share the source and original SD protocol")


def collect(manifest, root, baseline_root, refs, score):
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    mode = manifest["mode"]
    if (mode not in ("development","holdout") or manifest["models"] != ["qwen"] or
        manifest["case_count"] != (200 if mode == "development" else 800) or
        manifest["case_ids"] != manifest["development_case_ids" if mode == "development" else "holdout_case_ids"] or
        manifest["selection_policy"] != "decision_proper_where_v1" or
        manifest["question_overlap_policy"] != "proper_where" or manifest["entity_reserve_fraction"] != .25 or
        manifest["keep_fraction"] != .1 or manifest["student_rollout_tokens"] != 8 or
        manifest["gain_head_mode"] != "kv" or manifest["fixed_value_learning_rate"] != .0001 or
        manifest["max_answer_steps"] != 8 or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The fixed question-overlap protocol changed")
    rows, errors, invalid, unmatched = [], [], [], []
    for case in manifest["case_ids"]:
        folder = Path(root) / "cases/qwen" / case
        path, old_path = folder / "result.json", Path(baseline_root) / "cases/qwen" / case / "result.json"
        if not path.exists():
            if (folder / "error.json").exists():
                errors.append(dict(case=case,type=json.loads((folder / "error.json").read_text()).get("type")))
            continue
        result, ref = json.loads(path.read_text()), refs[case]
        try:
            validate_candidate(result,case,"qwen",ref["task"],gain_head_mode="kv",
                               selection_policy="decision_proper_where_v1")
            training = result["training"]
            if (result["selection"]["question_overlap_policy"] != "proper_where" or
                training["value_learning_rate"] != .0001 or training["gain_learning_rate"] != .01 or
                training["adaptive_max_answer_steps"] != 8 or training["adaptive_answer_KL_stop"] != .5 or
                len(training["steps"]) != 2 + result["self_study"]["verified"] or
                training.get("student_question_matches_teacher") or
                result["standalone_cartridge"].get("answer_format") != "original"):
                raise ValueError("The original-SD configuration changed")
        except (ValueError, KeyError, TypeError) as exc:
            invalid.append(dict(case=case,reason=str(exc)));continue
        if not old_path.exists():
            unmatched.append(dict(case=case,reason="baseline_missing"));continue
        try:
            old = json.loads(old_path.read_text());validate_old(old,result,case)
        except (ValueError, KeyError, TypeError) as exc:
            unmatched.append(dict(case=case,reason=str(exc)));continue
        grade = lambda text:int(score(text,ref["answers"],"babilong",ref["task"])[0])
        pair, old_pair = result["standalone_cartridge"]["original_question_controls"], old["standalone_cartridge"]["paired_prompt_control"]
        rows.append(dict(case=case,task=ref["task"],length=result["length"],
            whole=grade(result["self_judge"]["native_prediction"]),zero=grade(pair["zero"]["prediction"]),
            teacher=grade(result["teacher"]["prediction"]),cartridge=grade(result["prediction"]),
            old_whole=grade(old["self_judge"]["native_prediction"]),old_zero=grade(old_pair["zero_original"]["prediction"]),
            old_teacher=grade(old["teacher"]["prediction"]),old_cartridge=grade(old_pair["trained_original"]["prediction"]),
            selection_changed=(old["selection"]["selected_original_chunks"] != result["selection"]["selected_original_chunks"]),
            same_whole_output=old["self_judge"]["native_prediction"] == result["self_judge"]["native_prediction"],
            same_teacher_tokens=old["teacher"]["tokens"] == result["teacher"]["tokens"],
            same_zero_tokens=old_pair["zero_original"]["tokens"] == pair["zero"]["tokens"],
            same_cartridge_tokens=old_pair["trained_original"]["tokens"] == result["generated_token_ids"]))
    subsets = defaultdict(list)
    for row in rows:
        subsets["task:"+row["task"]].append(row);subsets["length:"+str(row["length"])].append(row)
        subsets["selection_changed:"+str(row["selection_changed"]).lower()].append(row)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),mode=mode,model="qwen",
        expected=manifest["case_count"],metric="answer_focused_lexical_v2",cached_scores=True,online_api_calls=0,
        expected_development_changed_selections=manifest["expected_development_changed_selections"],
        reference_answers_used_only_posthoc=True,overall=summarize(rows),
        groups={name:summarize(value) for name,value in sorted(subsets.items())},
        rows=rows,errors=errors,invalid=invalid,unmatched=unmatched)


def render(report):
    s = report["overall"];a = s["accuracy"]
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    return "\n".join(["# Qwen: question-guided selection and original SD", "",
        "| Cases | Whole | Old Teacher | Old cartridge | New Teacher | New zero | New cartridge |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {s['n']} | {fmt(a['whole'])} | {fmt(a['old_teacher'])} | {fmt(a['old_cartridge'])} | "
        f"{fmt(a['teacher'])} | {fmt(a['zero'])} | {fmt(a['cartridge'])} |", "",
        f"Cartridge vs its Teacher: {fmt(s['cartridge_vs_teacher']['delta_pp'])} pp; "
        f"vs old cartridge: {fmt(s['cartridge_vs_old']['delta_pp'])} pp. "
        f"Selection changed on {s['changed_selection']} paired cases; see the unchanged-selection group for "
        "cross-run token drift.", "",
        "The Teacher uses its original generic output requirement. Both cartridges and both zero-update controls "
        "use the original question. The selector uses question text and cached scores only; labels are used "
        "after generation by the unified loose lexical scorer. The candidate was designed using development results.", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","candidate-root","baseline-root","references","scorer-dir","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.scorer_dir))
    from report_hybrid_partial import BABILONG_SCORER_VERSION,score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":
        raise ValueError("Unexpected scorer version")
    result = collect(json.loads(args.manifest.read_text()),args.candidate_root,args.baseline_root,
                     json.loads(args.references.read_text()),score_prediction)
    for suffix,text in ((".json",json.dumps(result,indent=2)+"\n"),(".md",render(result))):
        path=args.output.with_suffix(suffix);temporary=path.with_suffix(path.suffix+".tmp")
        temporary.write_text(text);temporary.replace(path)


if __name__ == "__main__":
    main()
