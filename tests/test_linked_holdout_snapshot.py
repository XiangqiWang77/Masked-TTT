import pytest

from experiments.babilong_linked_selection.snapshot_holdout import freeze


def fixture_reports():
    def cohort(prefix, count):
        return [f"babilong-{length}-qa1-{prefix}{i:03d}" for i in range(count) for length in (8, 16, 32, 64)]

    development = dict(case_ids=cohort("d", 50))
    keys = cohort("h", 200)
    holdout = dict(mode="holdout", case_count=800, case_ids=keys, models=["qwen", "gpt"], keep_fraction=.1,
        selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25, basis_source="document", reference_answers_used_in_inference=False,
        online_api_calls=0, selected_transfer_weight=0, holdout_evaluated_at_selection=False,
        qualitative_case_sources=["qa1-h000", "qa1-h001"])
    row = dict(case=keys[8], task="qa1", length=8192, whole=1, baseline_source="full_1000",
        same_full_answer=True, baseline10=0, zero=0, teacher=1, cartridge=1,
        zero_teacher_format=1, cartridge_teacher_format=1)
    report = dict(metric="answer_focused_lexical_v2", development_only=False,
        expected_per_model=800, selection_policy="decision_entity_link_v1", selected_fraction=.1,
        baseline_fraction=.1, cached_scores=True, online_api_calls=0,
        evaluation_answers_used_only_posthoc=True, updated=0, errors=[], invalid=[],
        models=dict(qwen=dict(rows=[row]), gpt=dict(rows=[])))
    return holdout, development, report


def test_partial_holdout_snapshot_preserves_baseline_and_excludes_case_ids():
    holdout, development, report = fixture_reports()
    result = freeze(holdout, development, report)
    assert result["models"]["qwen"]["overall"]["n"] == 1
    assert result["models"]["qwen"]["overall"]["pairs"]["selection_and_method"]["delta_pp"] == 100
    assert result["models"]["gpt"]["overall"]["n"] == 0
    assert result["models"]["qwen"]["previously_viewed_cases"] == 0
    assert "rows" not in result["models"]["qwen"]


@pytest.mark.parametrize("change", ["source_leak", "training_leak", "missing_score", "duplicate"])
def test_holdout_snapshot_rejects_invalid_pairing(change):
    holdout, development, report = fixture_reports()
    if change == "source_leak":
        holdout["case_ids"][0] = development["case_ids"][0]
    elif change == "training_leak":
        holdout["reference_answers_used_in_inference"] = True
    elif change == "missing_score":
        report["models"]["qwen"]["rows"][0]["cartridge"] = None
    else:
        report["models"]["qwen"]["rows"].append(report["models"]["qwen"]["rows"][0])
    with pytest.raises(ValueError):
        freeze(holdout, development, report)
