import pytest

from experiments.value_lr_physical.freeze_calibration_ceiling import FAMILIES, freeze, summarize


def fixture_reports():
    keys = [f"babilong-{length}-qa1-{source:03d}" for source in range(50) for length in (8, 16, 32, 64)]
    protocols = dict(qwen="q", gpt="g")
    common = dict(case_ids=keys, model_protocols=protocols, keep_fraction=.1,
                  basis_source="document", selected_transfer_weight=0)
    value_manifest = dict(common, reference_answers_used=False, value_learning_rates=[.0001, .0003, .001])
    step_manifest = dict(common, reference_answers_used_in_inference=False, step_budgets=[2, 4, 8])
    def rows(family):
        return [dict(case=key, task="qa1", length=8192, teacher=0, whole=0, zero=0,
                     **{arm: int(i % 3 == j) for j, arm in enumerate(FAMILIES[family])})
                for i, key in enumerate(keys)]
    common_report = dict(metric="answer_focused_lexical_v2", development_only=True,
                         online_api_calls=0, errors=[], invalid=[])
    value_report = dict(common_report, same_teacher_student_kl=True, same_document_basis=True,
        evaluation_answers_used_in_training=False,
        models=dict(qwen=dict(rows=rows("value_learning_rate")), gpt=dict(rows=rows("value_learning_rate"))))
    step_report = dict(common_report, same_trajectory=True, evaluation_answers_used_only_posthoc=True,
        models=dict(qwen=dict(rows=rows("update_steps")), gpt=dict(rows=rows("update_steps"))))
    return value_manifest, value_report, step_manifest, step_report


def test_oracle_uses_gold_only_as_posthoc_upper_bound():
    values, value_report, steps, step_report = fixture_reports()
    result = freeze(values, value_report, steps, step_report)
    stats = result["models"]["qwen"]["value_learning_rate"]["overall"]
    assert stats["oracle_accuracy"] == 100 and stats["answer_changes_across_candidates"] == 200
    assert stats["oracle_above_teacher_pp"] == 100
    assert "not deployable" in result["oracle_definition"]
    assert "rows" not in result["models"]["qwen"]["value_learning_rate"]
    assert summarize([], FAMILIES["update_steps"])["oracle_accuracy"] is None


@pytest.mark.parametrize("change", ["duplicate", "missing_score", "training_leak", "different_cohort"])
def test_oracle_rejects_incomplete_or_unmatched_data(change):
    values, value_report, steps, step_report = fixture_reports()
    if change == "duplicate":
        value_report["models"]["qwen"]["rows"][-1] = value_report["models"]["qwen"]["rows"][0]
    elif change == "missing_score":
        step_report["models"]["gpt"]["rows"][0]["step8"] = None
    elif change == "training_leak":
        values["reference_answers_used"] = True
    else:
        steps["case_ids"] = steps["case_ids"][:-1]
    with pytest.raises(ValueError):
        freeze(values, value_report, steps, step_report)
