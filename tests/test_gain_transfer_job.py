import pytest

from experiments.value_lr_physical.run_gain_transfer import preflight, validate_result
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_loaded_model_preflight_checks_observation_and_release(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    initial = engine.config
    result = preflight(engine)
    assert result["passed"] and result["gain_observer_neutral"]
    assert result["full_and_source_released"] and result["readout_available"]
    assert engine.config is initial


def valid_result():
    return dict(
        training=dict(objective="on-policy KL(teacher || student)", value_basis_source="document",
            selected_transfer_weight=0, student_rollout_limit=8, teacher_conditioned_on_student_prefix=True,
            student_rollouts_refreshed_each_step=True, steps=[dict(gain_gradient_norm=2.0)]),
        selection=dict(source_case_key="case", reference_answers_used=False, online_api_calls=0),
        self_study=dict(reference_answers_used=False, student_answer_hint_used=False),
        teacher_answer_in_final_prompt=False, backbone_unchanged=True, original_document_KV_unchanged=True,
        memory=dict(full_cache_release_before_standalone=dict(full_cache_tensors_live_before_standalone=0)),
        generated_token_ids=[1, 2],
        standalone_cartridge=dict(generated_token_ids=[1, 2], memory=dict(kv_only_materialized=True,
            cartridge_adapter_bytes=0, additional_source_kv_bytes_before_decode=0,
            additional_training_adapter_bytes_before_decode=0)),
    )


@pytest.mark.parametrize("change", ["transfer_loss", "leak", "retained_cache", "modified_model", "wrong_gradient"])
def test_result_gate_rejects_method_or_observation_changes(change):
    result = valid_result()
    observations = [dict(gradient=dict(total_norm=2.0))]
    validate_result(result, observations, {"key": "case"})
    if change == "transfer_loss":
        result["training"]["selected_transfer_weight"] = .1
    elif change == "leak":
        result["selection"]["reference_answers_used"] = True
    elif change == "retained_cache":
        result["standalone_cartridge"]["memory"]["additional_training_adapter_bytes_before_decode"] = 8
    elif change == "modified_model":
        result["backbone_unchanged"] = False
    else:
        observations[0]["gradient"]["total_norm"] = 1.0
    with pytest.raises(ValueError):
        validate_result(result, observations, {"key": "case"})
