"""Choose one SD step budget per model from the full development split only."""
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.long_rollout_steps.report import summarize, validate_manifest


def development_choices(manifest, report):
    if validate_manifest(manifest) != 200:
        raise ValueError("Checkpoint selection requires the full 200-case development split")
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    if (report["expected_per_model"] != 200 or report["metric"] != "answer_focused_lexical_v2" or
        not report["cached_scores"] or report["online_api_calls"] != 0 or
        not report["evaluation_answers_used_only_posthoc"] or report["errors"] or report["invalid"] or
        set(report["models"]) != set(manifest["models"])):
        raise ValueError("A clean complete checkpoint development report is required")
    decisions = {}
    for model in manifest["models"]:
        item = report["models"][model]
        rows = item["rows"]
        if len(rows) != 200 or {row["case"] for row in rows} != set(manifest["case_ids"]):
            raise ValueError("The development cases are incomplete for " + model)
        if item["overall"] != summarize(rows):
            raise ValueError("Stored development statistics do not match the paired rows")

        def criteria(step):
            key = "step" + str(step)
            correct = sum(row[key] for row in rows)
            regressions = sum(row["zero"] and not row[key] for row in rows)
            # Highest answer quality first; on ties prefer fewer regressions,
            # then fewer optimization steps.
            return (correct, -regressions, -step)

        chosen = max(manifest["step_budgets"], key=criteria)
        key = "step" + str(chosen)
        scores = {name: sum(row[name] for row in rows) for name in
                  ("whole", "zero", "teacher", "step2", "step4", "step8")}
        deltas = {name: (scores[key] - scores[name]) / 2 for name in ("whole", "zero", "teacher")}
        eligible = all(value > 0 for value in deltas.values())
        decisions[model] = dict(fixed_final_base_step=chosen, eligible=eligible,
            development_correct=scores, delta_pp=deltas,
            tie_break="fewer regressions against zero-update, then fewer steps",
            references_used_only_after_development_generation=True)
    return decisions
