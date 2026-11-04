from copy import deepcopy

import pytest

from experiments.value_lr_physical.freeze_readout_summary import freeze


def report_fixture():
    rows = []
    for i in range(10):
        for length in (8, 16, 32, 64):
            rows.append(dict(
                case=f"babilong-{length}-qa{i}-300000",
                whole=1, selected=0, group="whole_only", cartridge=0,
                probe=dict(prefix="same original question; before first answer token",
                    kl_full_native_to_selected_native=0.3,
                    kl_full_native_to_selected_adapted=0.2,
                    kl_full_native_to_full_adapted=0.25,
                    kl_selected_native_to_selected_adapted=0.5,
                    reference_answers_used=False, teacher_answer_in_student_prompt=False,
                    full_cache_or_model_modified=False),
            ))
    return dict(scope="fixed 40-case development diagnostic; first answer position only",
        metric="full-vocabulary first-token KL on the same original question",
        references_used_only_posthoc_for_grouping=True, model_or_cache_modified=False,
        teacher_answer_in_student_prompt=False, online_api_calls=0,
        models=dict(qwen=dict(rows=rows, unavailable={}),
                    gpt=dict(rows=deepcopy(rows[:24]), unavailable={"not_recorded": 16})))


def test_snapshot_uses_paired_cases_and_omits_case_and_prediction_data():
    summary = freeze(report_fixture())
    assert summary["source_questions"] == 10
    assert summary["models"]["qwen"]["overall"]["n"] == 40
    assert summary["models"]["gpt"]["unavailable"] == {"not_recorded": 16}
    assert summary["models"]["gpt"]["overall"]["median_selected_over_full_effect"] == 2
    assert summary["models"]["qwen"]["overall"]["closer_to_full"] == 40
    assert "rows" not in summary["models"]["qwen"]


@pytest.mark.parametrize("change", ["duplicate", "wrong_length", "label_leak", "invalid_kl", "wrong_group", "scope"])
def test_snapshot_rejects_invalid_provenance_and_metrics(change):
    report = report_fixture()
    if change == "duplicate":
        report["models"]["qwen"]["rows"][-1] = deepcopy(report["models"]["qwen"]["rows"][0])
    elif change == "wrong_length":
        report["models"]["qwen"]["rows"][0]["case"] = "babilong-128-qa0-300000"
    elif change == "label_leak":
        report["models"]["qwen"]["rows"][0]["probe"]["reference_answers_used"] = True
    elif change == "invalid_kl":
        report["models"]["qwen"]["rows"][0]["probe"]["kl_full_native_to_full_adapted"] = float("nan")
    elif change == "wrong_group":
        report["models"]["gpt"]["rows"][0]["group"] = "both_correct"
    else:
        report["scope"] = "different task"
    with pytest.raises(ValueError):
        freeze(report)
