"""Fixed stronger Value rate, with the original full-Student SD path."""
from contextlib import contextmanager
from masked_ttt.config import TTTConfig


@contextmanager
def strong_value_runner(implementation, paired_controls, *, root, fixed_lr, physical_controls):
    if fixed_lr != .003:
        raise ValueError("The 0.003 candidate must be fixed before evaluation")
    old = (implementation.ROOT, implementation.replace, implementation.validate,
           implementation.validate_manifest, paired_controls.original_question_controls)
    old_root, old_replace, old_validate, old_manifest, old_controls = old

    def configured(value, **kwargs):
        if isinstance(value, TTTConfig) and "max_context" in kwargs:
            kwargs.update(value_learning_rate=fixed_lr, gain_head_mode="kv", steps=2,
                          max_answer_steps=2, align_student_teacher_prompt=False,
                          standalone_answer_format="original")
        return old_replace(value, **kwargs)

    def manifest_check(manifest):
        if (manifest.get("mode") not in ("development", "holdout") or
            manifest.get("case_count") != (200 if manifest["mode"] == "development" else 800) or
            manifest.get("models") != ["gpt"] or manifest.get("fixed_value_learning_rate") != fixed_lr or
            manifest.get("gain_head_mode") != "kv" or manifest.get("steps") != 2 or
            manifest.get("holdout_evaluated_at_selection") is not False):
            raise ValueError("The fixed stronger-Value original-SD protocol changed")
        return old_manifest(manifest)

    def validate(result, entry):
        old_validate(result, entry)
        training = result["training"]
        final = result["standalone_cartridge"]
        memory = final["memory"]
        practices = result["self_study"]["verified"]
        if (training["value_learning_rate"] != fixed_lr or training["gain_learning_rate"] != .01 or
            training["gain_head_mode"] != "kv" or len(training["steps"]) != 2 + practices or
            training.get("student_question_matches_teacher") or
            result["output_source"] != "standalone_cartridge" or
            not isinstance(result["self_judge"]["native_prediction"], str) or
            not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
            memory["additional_source_kv_bytes_before_decode"] != 0 or
            memory["additional_training_adapter_bytes_before_decode"] != 0):
            raise ValueError("The stronger-Value independent cartridge contract failed")
        training["fixed_value_learning_rate"] = fixed_lr

    implementation.ROOT = root
    implementation.replace = configured
    implementation.validate = validate
    implementation.validate_manifest = manifest_check
    paired_controls.original_question_controls = physical_controls
    try:
        yield
    finally:
        (implementation.ROOT, implementation.replace, implementation.validate,
         implementation.validate_manifest, paired_controls.original_question_controls) = old
