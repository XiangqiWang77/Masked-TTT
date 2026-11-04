import json

import pytest

from experiments.value_lr_physical.report_gain_transfer import collect, write_report


def component(common, contrast):
    squared = common * common + contrast * contrast
    return dict(selected_chunks=2, common_norm=common, contrast_norm=contrast,
                total_norm=squared ** .5, common_squared_fraction=common * common / squared if squared else None)


def make_case(key):
    return dict(key=key, model="qwen", task="qa1",
        training=dict(objective="on-policy KL(teacher || student)", selected_transfer_weight=0,
                      steps=[dict(gain_gradient_norm=2.0)]),
        selection=dict(source_case_key=key, reference_answers_used=False, online_api_calls=0),
        checks=dict(backbone_unchanged=True, original_document_KV_unchanged=True,
                    full_cache_released=True, teacher_answer_in_final_prompt=False),
        memory=dict(kv_only_materialized=True, additional_source_kv_bytes_before_decode=0,
                    additional_training_adapter_bytes_before_decode=0),
        observations=[dict(gradient=component(1.2, 1.6), optimizer_update=component(0, .5),
                           final_gain=component(1, 0))],
        readout=dict(kl_full_native_to_selected_native=.3, kl_full_native_to_selected_adapted=.2,
                     kl_full_native_to_full_adapted=.25, kl_selected_native_to_selected_adapted=.5,
                     reference_answers_used=False, teacher_answer_in_student_prompt=False,
                     full_cache_or_model_modified=False))


def test_partial_report_counts_available_cases_and_excludes_answers(tmp_path):
    keys = [f"case-{i}" for i in range(40)]
    (tmp_path / "manifest.json").write_text(json.dumps(dict(case_ids=keys, models=["qwen", "gpt"],
        reference_answers_used=False, online_api_calls=0, selected_transfer_weight=0, diagnosis_only=True,
        scope="probe")))
    folder = tmp_path / "cases/qwen"
    folder.mkdir(parents=True)
    row = make_case(keys[0])
    row["prediction"] = "private answer"
    (folder / (keys[0] + ".json")).write_text(json.dumps(row))
    report = write_report(tmp_path)
    summary = report["models"]["qwen"]["overall"]
    assert summary["cases"] == 1 and summary["first_steps"] == 1
    assert summary["median_first_gradient_common_squared_fraction"] == pytest.approx(.36)
    assert summary["median_first_update_common_squared_fraction"] == 0
    assert summary["median_selected_over_full_first_token_change"] == 2
    assert report["models"]["gpt"]["overall"]["cases"] == 0
    assert "private answer" not in (tmp_path / "latest.json").read_text()
    row["checks"]["full_cache_released"] = False
    (folder / (keys[0] + ".json")).write_text(json.dumps(row))
    with pytest.raises(ValueError):
        collect(tmp_path)
