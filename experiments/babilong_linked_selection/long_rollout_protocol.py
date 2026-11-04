"""Paired evaluation rules for linked 10% evidence plus longer SD rollouts.

Answers enter only post-generation scoring. The development rule is frozen
before this combined variant is run, and its 800-case holdout is separated by
source question rather than only by document length.
"""
import math
import statistics


TEACHER_INSTRUCTION = "Return only the final answer in the required format. Do not add an explanation."


def validate_partition(development, holdout):
    """Require four lengths of each source question in one split only."""
    def grouped(cases):
        groups = {}
        for case in cases:
            parts = case.split("-", 2)
            if len(parts) != 3 or parts[0] != "babilong" or parts[1] not in ("8", "16", "32", "64"):
                raise ValueError("Unexpected BABILONG case identifier")
            groups.setdefault(parts[2], set()).add(parts[1])
        if any(lengths != {"8", "16", "32", "64"} for lengths in groups.values()):
            raise ValueError("Each source question needs all four lengths")
        return groups

    if (len(development) != 200 or len(set(development)) != 200 or
        len(holdout) != 800 or len(set(holdout)) != 800 or set(development) & set(holdout)):
        raise ValueError("The 200/800 case split changed")
    dev, held = grouped(development), grouped(holdout)
    if len(dev) != 50 or len(held) != 200 or set(dev) & set(held):
        raise ValueError("Source questions must be disjoint across the two splits")
    if not {"qa1-300023", "qa2-300037"}.issubset(held):
        raise ValueError("The two previously viewed source questions must remain marked")


def prepare_development_manifest(rollout, linked, profile):
    """Freeze the combination before generating or scoring any combined cases."""
    development, holdout = rollout["case_ids"], rollout["holdout_case_ids"]
    validate_partition(development, holdout)
    if (development != linked["case_ids"] or holdout != linked["holdout_case_ids"] or
        rollout["model_protocols"] != linked["model_protocols"] or
        rollout["mode"] != "development" or rollout["case_count"] != 200 or
        rollout["models"] != ["qwen", "gpt"] or
        rollout["selection_policy"] != "cached_decision_score10" or
        linked["selection_policy"] != "decision_entity_link_v1" or
        linked["entity_reserve_fraction"] != .25 or
        rollout["keep_fraction"] != .1 or linked["keep_fraction"] != .1 or
        rollout["student_rollout_tokens"] != 32 or
        rollout["teacher_instruction"] != TEACHER_INSTRUCTION or
        rollout["verifier_policy"] != "original_exact" or
        rollout["basis_source"] != "document" or linked["basis_source"] != "document" or
        rollout["selected_transfer_weight"] != 0 or linked["selected_transfer_weight"] != 0 or
        rollout["reference_answers_used_in_inference"] is not False or
        linked["reference_answers_used_in_inference"] is not False or
        rollout["online_api_calls"] != 0 or linked["online_api_calls"] != 0 or
        profile["teacher_response_instruction"] != TEACHER_INSTRUCTION or
        profile["student_rollout_tokens"] != 8 or profile["steps"] != 2 or
        profile["gain_learning_rate"] != .01 or profile["value_learning_rate"] != .0001):
        raise ValueError("The two frozen development protocols or original OLMo profile differ")
    return dict(rollout, scope="Linked 10% evidence plus 32-token original-method SD development",
        selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25,
        development_case_ids=list(development), holdout_case_ids=list(holdout),
        selection_reason="Combine cached-score evidence links with a longer student prefix; "
                         "same selected-teacher/full-student SD and independent KV materialization.",
        development_gate="Beat own whole, selected zero and Teacher on all 200 cases; "
                         "report the linked 8-token comparison separately.",
        holdout_gate="At least +3 pp versus own Teacher and positive versus whole and selected zero; "
                     "report 792 cases excluding two previously viewed sources.",
        combined_development_results_seen=False)


def prepare_holdout_manifest(development, report):
    """Promote only models that passed the frozen combined-development gate."""
    validate_partition(development["development_case_ids"], development["holdout_case_ids"])
    if (development["mode"] != "development" or development["case_count"] != 200 or
        report["mode"] != "development" or report["expected_per_model"] != 200 or
        report["metric"] != "answer_focused_lexical_v2" or
        report["selection_policy"] != "decision_entity_link_v1" or not report["cached_scores"] or
        report["student_rollout_tokens"] != 32 or report["selected_transfer_weight"] != 0 or
        not report["evaluation_answers_used_only_posthoc"] or report["online_api_calls"] != 0 or
        report["errors"] or report["invalid"]):
        raise ValueError("A clean, fully matched development report is required")
    decisions = {}
    for model in development["models"]:
        item = report["models"][model]
        rows = item["rows"]
        if len(rows) != 200:
            raise RuntimeError("Waiting for the full development: " + model)
        if ({row["case"] for row in rows} != set(development["development_case_ids"]) or
            summarize(rows) != item["overall"]):
            raise ValueError("Development scores do not match the frozen per-case predictions")
        decision = development_decision(item["overall"])
        if decision["reason"] == "development_incomplete":
            raise RuntimeError("Waiting for the full development: " + model)
        if decision != item["decision"]:
            raise ValueError("The stored development decision does not match the scores")
        decisions[model] = decision
    selected = [model for model in development["models"] if decisions[model]["eligible"]]
    return dict(development, scope="Linked 10% plus 32-token SD source-disjoint holdout",
        case_ids=list(development["holdout_case_ids"]), case_count=800, mode="holdout",
        models=selected, development_decisions=decisions, combined_development_results_seen=True,
        holdout_evaluated_at_selection=False)


def validate_linked_baseline(case, old, new):
    """Require the same teacher/evidence/zero state and matching first prefix."""
    previous = old["standalone_cartridge"]["paired_prompt_control"]
    current = new["standalone_cartridge"]["original_question_controls"]
    older_training, newer_training = old["training"], new["training"]
    older_selection, newer_selection = old["selection"], new["selection"]
    fields = ("question", "answer", "evidence", "source_chunk", "verifier_answer", "accepted")

    def proposals(result):
        return [{key: item[key] for key in fields} for item in result["self_study"]["checks"]]

    if not older_training["steps"] or not newer_training["steps"]:
        raise ValueError("Both runs need at least one SD update")
    older_prefix = older_training["steps"][0]["student_rollout_tokens"]
    newer_prefix = newer_training["steps"][0]["student_rollout_tokens"]
    shared_fields = ("objective", "optimizer", "teacher_cap", "teacher_target_scope",
                     "adaptive_max_answer_steps", "adaptive_answer_KL_stop")
    if (old["key"] != case or new["key"] != case or old["task"] != new["task"] or
        old["model"] != new["model"] or older_selection["source_case_key"] != case or
        newer_selection["source_case_key"] != case or
        older_selection["selection_policy"] != "decision_entity_link_v1" or
        newer_selection["selection_policy"] != "decision_entity_link_v1" or
        older_selection["entity_reserve_fraction"] != .25 or
        newer_selection["entity_reserve_fraction"] != .25 or
        older_selection["selected_original_chunks"] != newer_selection["selected_original_chunks"] or
        older_selection["total_chunks"] != newer_selection["total_chunks"] or
        older_selection["original_prompt_tokens"] != newer_selection["original_prompt_tokens"] or
        not older_selection["complete_original_document"] or not newer_selection["complete_original_document"] or
        older_selection["reference_answers_used"] or newer_selection["reference_answers_used"] or
        older_selection["online_api_calls"] != 0 or newer_selection["online_api_calls"] != 0 or
        old["teacher"]["tokens"] != new["teacher"]["tokens"] or
        not old["teacher"]["response_instruction_applied"] or
        not new["teacher"]["response_instruction_applied"] or
        older_training["student_rollout_limit"] != 8 or newer_training["student_rollout_limit"] != 32 or
        older_training["selected_transfer_weight"] != 0 or newer_training["selected_transfer_weight"] != 0 or
        older_training.get("value_basis_source", "document") != "document" or
        newer_training["value_basis_source"] != "document" or
        any(older_training.get(key) != newer_training.get(key) for key in shared_fields) or
        older_training["steps"][0]["question_path"] != "main" or
        newer_training["steps"][0]["question_path"] != "main" or
        not older_prefix or newer_prefix[:len(older_prefix)] != older_prefix or
        previous["zero_original"]["tokens"] != current["zero"]["tokens"] or
        proposals(old) != proposals(new) or
        not previous["same_selected_kv"] or not previous["same_generation_limit"] or
        not current["same_selected_kv"] or not current["same_generation_limit"] or
        not current["same_question_suffix"] or
        previous["reference_answers_used"] or current["reference_answers_used"] or
        previous["teacher_answer_in_student_prompt"] or current["teacher_answer_in_student_prompt"]):
        raise ValueError("The linked 8- and 32-token runs are not directly paired")


def summarize(rows):
    """All controls are same-run; linked8 uses the strictly matched subset."""
    names = ("whole", "zero", "teacher", "cartridge")
    if any(any(type(row[key]) not in (bool, int) or row[key] not in (0, 1) for key in names)
           for row in rows):
        raise ValueError("Expected binary post-generation scores")
    paired = [row for row in rows if row.get("linked8") is not None]
    if any(type(row["linked8"]) not in (bool, int) or row["linked8"] not in (0, 1) for row in paired):
        raise ValueError("Expected binary linked baseline scores")

    def rate(items, key):
        return 100 * sum(row[key] for row in items) / len(items) if items else None

    def change(items, previous):
        gains = sum(row["cartridge"] and not row[previous] for row in items)
        losses = sum(row[previous] and not row["cartridge"] for row in items)
        return {"n": len(items), "new_correct": gains, "new_wrong": losses,
                "delta_pp": 100 * (gains - losses) / len(items) if items else None}

    ratios = [row["cartridge_percent"] for row in rows if row.get("cartridge_percent") is not None]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("Invalid cartridge storage ratio")
    return {"n": len(rows), "accuracy": {key: rate(rows, key) for key in names},
            "versus_whole": change(rows, "whole"), "versus_zero": change(rows, "zero"),
            "versus_teacher": change(rows, "teacher"),
            "paired_linked8_accuracy": rate(paired, "linked8"),
            "paired_cartridge_accuracy": rate(paired, "cartridge"),
            "versus_linked8": change(paired, "linked8"),
            "median_cartridge_percent": statistics.median(ratios) if ratios else None}


def development_decision(summary, required=200, minimum_matched=180):
    """Fixed primary gate uses the same-run Teacher, full, and zero controls."""
    if summary["n"] != required:
        return {"eligible": False, "reason": "development_incomplete"}
    accuracy = summary["accuracy"]
    deltas = {key: accuracy["cartridge"] - accuracy[key] for key in ("whole", "zero", "teacher")}
    if any(not math.isfinite(value) for value in deltas.values()):
        raise ValueError("Invalid development score")
    passed = all(value > 0 for value in deltas.values())
    return {"eligible": passed, "reason": "beats_all_development_controls" if passed else
            "does_not_beat_all_development_controls", "delta_pp": deltas,
            "linked8_comparison": linked8_comparison(summary, minimum_matched)}


def linked8_comparison(summary, minimum_matched):
    """Keep attribution to the longer prefix separate from primary quality."""
    paired = summary["versus_linked8"]
    enough = paired["n"] >= minimum_matched
    return {"available": enough, "matched": paired["n"], "minimum_matched": minimum_matched,
            "delta_pp": paired["delta_pp"], "improved": paired["delta_pp"] > 0 if enough else None}


def holdout_decision(summary, required=792, minimum_matched=713):
    """Apply the 3-point target excluding two previously viewed source questions."""
    decision = development_decision(summary, required, minimum_matched)
    if "delta_pp" not in decision:
        return decision
    passed = decision["eligible"] and decision["delta_pp"]["teacher"] >= 3 - 1e-9
    decision["eligible"] = passed
    decision["reason"] = "holdout_target_met" if passed else "holdout_target_not_met"
    return decision
