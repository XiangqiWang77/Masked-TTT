"""Enforce fixed Value LR, original selection and physically independent KV."""
from contextlib import contextmanager

from masked_ttt.config import TTTConfig


@contextmanager
def physical_value_runner(implementation, paired_controls, *, root, fixed_lr, physical_controls):
    if fixed_lr not in (.0001, .0003, .001):
        raise ValueError("Value LR must come from the predeclared development sweep")
    original = (implementation.ROOT, implementation.replace, implementation.validate,
                implementation.validate_manifest, paired_controls.original_question_controls)
    old_root, old_replace, old_validate, old_manifest, old_controls = original

    def configured(value, **kwargs):
        if isinstance(value, TTTConfig) and "max_context" in kwargs:
            kwargs.update(value_learning_rate=fixed_lr, gain_head_mode="kv")
        return old_replace(value, **kwargs)

    def validate_manifest(manifest):
        if (manifest.get("mode") != "holdout" or manifest.get("case_count") != 800 or
            manifest.get("fixed_value_learning_rate") != fixed_lr or manifest.get("gain_head_mode") != "kv" or
            manifest.get("holdout_evaluated_at_selection") is not False):
            raise ValueError("The development-fixed physical holdout changed")
        return old_manifest(manifest)

    def validate(result, entry):
        old_validate(result, entry)
        training = result["training"]
        final = result["standalone_cartridge"]
        memory = final["memory"]
        practices = result["self_study"]["verified"]
        if (training["gain_head_mode"] != "kv" or len(training["steps"]) != 2 + practices or
            result["output_source"] != "standalone_cartridge" or
            result["generated_token_ids"] != final["generated_token_ids"] or
            not isinstance(result["self_judge"]["native_prediction"], str) or
            not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
            memory["additional_source_kv_bytes_before_decode"] != 0 or
            memory["additional_training_adapter_bytes_before_decode"] != 0):
            raise ValueError("The fixed-LR independent-cartridge contract failed")
        training["fixed_value_learning_rate"] = fixed_lr

    implementation.ROOT = root
    implementation.replace = configured
    implementation.validate = validate
    implementation.validate_manifest = validate_manifest
    paired_controls.original_question_controls = physical_controls
    try:
        yield
    finally:
        (implementation.ROOT, implementation.replace, implementation.validate,
         implementation.validate_manifest, paired_controls.original_question_controls) = original
