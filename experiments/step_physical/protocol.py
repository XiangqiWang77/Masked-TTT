"""Freeze the original-selector step winner using completed development only."""
import math

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition


def development_choice(manifest, report, model):
    development, holdout = manifest["case_ids"], manifest["holdout_case_ids"]
    validate_partition(development, holdout)
    if (model not in ("qwen", "gpt") or model not in manifest["models"] or
        manifest["mode"] != "development" or manifest["case_count"] != 200 or
        manifest["keep_fraction"] != .1 or manifest["step_budgets"] != [2, 4, 8] or
        manifest["basis_source"] != "document" or manifest["selected_transfer_weight"] != 0 or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        report["metric"] != "answer_focused_lexical_v2" or report["expected_per_model"] != 200 or
        not report["development_only"] or report["selected_fraction"] != .1 or
        report["step_budgets"] != [2, 4, 8] or not report["same_trajectory"] or
        not report["cached_scores"] or report["online_api_calls"] != 0 or
        not report["evaluation_answers_used_only_posthoc"] or
        any(item.get("model") == model for item in report["errors"] + report["invalid"])):
        raise ValueError("The original-selector step development protocol changed")
    item = report["models"][model]
    rows = item["rows"]
    names = ("whole", "zero", "teacher", "step2", "step4", "step8")
    if (len(rows) != 200 or {row["case"] for row in rows} != set(development) or
        any(type(row[key]) is not int or row[key] not in (0, 1) for row in rows for key in names)):
        raise ValueError("The selected model needs 200 complete paired development cases")
    correct = {name: sum(row[name] for row in rows) for name in names}
    if item["overall"]["n"] != 200 or any(not math.isclose(
        item["overall"]["accuracy"][name], correct[name] / 2, abs_tol=1e-9) for name in names):
        raise ValueError("The stored accuracy does not match its paired rows")
    threshold = max(correct[name] for name in ("whole", "zero", "teacher", "step2"))
    eligible = [step for step in (4, 8) if correct["step" + str(step)] > threshold]
    if not eligible:
        expected = dict(steps=None, reason="no_step_budget_beats_all_controls")
    else:
        def criteria(step):
            key = "step" + str(step)
            regressions = sum(row["step2"] and not row[key] for row in rows)
            return correct[key], -regressions, -step
        best = max(eligible, key=criteria)
        expected = dict(steps=best, reason="predeclared_development_rule",
                        development_accuracy=correct["step" + str(best)] / 2,
                        holdout_evaluated=False)
    if item["choice"] != expected:
        raise ValueError("The predeclared step choice differs from the paired rows")
    return expected


def holdout_manifest(development, report, profile, model):
    choice = development_choice(development, report, model)
    if choice["steps"] is None:
        return None
    instruction = "Return only the final answer in the required format. Do not add an explanation."
    if (profile["student_rollout_tokens"] != 8 or profile["steps"] != 2 or
        profile["value_learning_rate"] != .0001 or profile["gain_learning_rate"] != .01 or
        profile["teacher_response_instruction"] != instruction):
        raise ValueError("The original OLMo profile changed")
    return dict(scope="Physical cartridge holdout with development-fixed original-SD step budget",
        mode="holdout", case_count=800, case_ids=list(development["holdout_case_ids"]),
        development_case_ids=list(development["case_ids"]), holdout_case_ids=list(development["holdout_case_ids"]),
        models=[model], model_protocols=development["model_protocols"], keep_fraction=.1,
        selection_policy="cached_decision_score10", teacher_instruction=instruction,
        verifier_policy="original_exact", student_rollout_tokens=8, basis_source="document",
        fixed_value_learning_rate=.0001, gain_head_mode="kv", steps=choice["steps"],
        selected_transfer_weight=0., reference_answers_used_in_inference=False,
        qualitative_case_sources=["qa1-300023", "qa2-300037"], online_api_calls=0,
        random_order_seed=development["random_order_seed"], max_gpu_hours=36,
        development_choice=choice, holdout_evaluated_at_selection=False)
