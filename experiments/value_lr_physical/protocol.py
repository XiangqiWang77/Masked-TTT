"""Select a Value learning rate from completed, source-disjoint development."""
import math

from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition


def name_for_lr(value):
    return "lr_" + f"{value:.4f}"


def development_choice(manifest, report, model):
    if model not in ("qwen", "gpt"):
        raise ValueError("Unsupported model")
    development, holdout = manifest["case_ids"], manifest["holdout_case_ids"]
    validate_partition(development, holdout)
    rates = manifest["value_learning_rates"]
    if (manifest["case_count"] != 200 or manifest["holdout_count"] != 800 or
        rates != [.0001, .0003, .001] or manifest["keep_fraction"] != .1 or
        manifest["gain_learning_rate"] != .01 or manifest["basis_source"] != "document" or
        manifest["selected_transfer_weight"] != 0 or manifest["reference_answers_used"] is not False or
        report["metric"] != "answer_focused_lexical_v2" or report["expected"] != 200 or
        not report["development_only"] or report["evaluation_answers_used_in_training"] or
        report["online_api_calls"] != 0 or not report["same_teacher_student_kl"] or
        not report["same_document_basis"] or report["selected_transfer_weight"] != 0 or
        report["rates"] != rates):
        raise ValueError("The fixed original-SD development protocol changed")
    if any(item.get("model") == model for item in report["errors"] + report["invalid"]):
        raise ValueError("The requested model has invalid or failed paired cases")
    item = report["models"][model]
    rows = item["rows"]
    if len(rows) != 200 or {row["case"] for row in rows} != set(development):
        raise ValueError("This model's development cases are incomplete")
    names = ["whole", "zero", "teacher"] + [name_for_lr(rate) for rate in rates]
    if any(type(row[key]) is not int or row[key] not in (0, 1)
           for row in rows for key in names):
        raise ValueError("Expected binary post-generation scores")
    correct = {key: sum(row[key] for row in rows) for key in names}
    if (item["overall"]["n"] != 200 or any(not math.isclose(
        item["overall"]["accuracy"][key], correct[key] / 2, abs_tol=1e-9) for key in names)):
        raise ValueError("The stored summary does not match its paired cases")
    eligible = [rate for rate in rates if correct[name_for_lr(rate)] >
                max(correct[name] for name in ("whole", "zero", "teacher"))]
    if not eligible:
        expected = dict(value_learning_rate=None, reason="none_exceeds_all_three_controls")
    else:
        def criteria(rate):
            key = name_for_lr(rate)
            wrong = sum(row["zero"] and not row[key] for row in rows)
            return correct[key], -wrong, -rate
        best = max(eligible, key=criteria)
        expected = dict(value_learning_rate=best, reason="predeclared_development_rule",
            development_accuracy=correct[name_for_lr(best)] / 2,
            control_threshold=max(correct[name] for name in ("whole", "zero", "teacher")) / 2,
            holdout_evaluated=False)
    if item["choice"] != expected:
        raise ValueError("The stored model choice does not match the predeclared rule")
    return expected


def holdout_manifest(development, report, profile, model):
    choice = development_choice(development, report, model)
    if choice["value_learning_rate"] is None:
        return None
    instruction = "Return only the final answer in the required format. Do not add an explanation."
    if (profile["student_rollout_tokens"] != 8 or profile["steps"] != 2 or
        profile["value_learning_rate"] != .0001 or profile["gain_learning_rate"] != .01 or
        profile["teacher_response_instruction"] != instruction):
        raise ValueError("The OLMo reference profile changed")
    return dict(scope="One-model physical cartridge holdout with development-fixed Value LR",
        mode="holdout", case_count=800, case_ids=list(development["holdout_case_ids"]),
        development_case_ids=list(development["case_ids"]), holdout_case_ids=list(development["holdout_case_ids"]),
        models=[model], model_protocols=development["model_protocols"], keep_fraction=.1,
        selection_policy="cached_decision_score10", teacher_instruction=instruction,
        verifier_policy="original_exact", student_rollout_tokens=8, basis_source="document",
        fixed_value_learning_rate=choice["value_learning_rate"], gain_head_mode="kv", steps=2,
        selected_transfer_weight=0., reference_answers_used_in_inference=False,
        qualitative_case_sources=["qa1-300023", "qa2-300037"], online_api_calls=0,
        random_order_seed=development["random_order_seed"], max_gpu_hours=36,
        development_choice=choice, holdout_evaluated_at_selection=False)
