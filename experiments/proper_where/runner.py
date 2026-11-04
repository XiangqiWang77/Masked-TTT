"""Use the fixed question-based selector without changing original SD."""
from contextlib import contextmanager
from masked_ttt.config import TTTConfig


@contextmanager
def proper_where_runner(implementation, runtime, controls, *, root, selector, physical_controls):
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.replace, runtime.FrozenJSONSelector, controls.original_question_controls)
    _, old_manifest, old_result, old_replace, _, _ = old

    def validate_manifest(manifest):
        if (manifest.get("mode") not in ("development", "holdout") or manifest.get("models") != ["qwen"] or
            manifest.get("selection_policy") != "decision_proper_where_v1" or
            manifest.get("question_overlap_policy") != "proper_where" or
            manifest.get("entity_reserve_fraction") != .25 or
            manifest.get("gain_head_mode") != "kv" or manifest.get("steps") != 2 or
            manifest.get("fixed_value_learning_rate") != .0001 or
            manifest.get("max_answer_steps") != 8 or manifest.get("student_rollout_tokens") != 8 or
            manifest.get("holdout_evaluated_at_selection") is not False):
            raise ValueError("The fixed question-overlap original-SD protocol changed")
        return old_manifest(dict(manifest, selection_policy="cached_decision_score10"))

    def configured(value, **kwargs):
        if isinstance(value, TTTConfig) and "max_context" in kwargs:
            kwargs.update(value_learning_rate=.0001, gain_head_mode="kv", steps=2,
                          max_answer_steps=8, align_student_teacher_prompt=False,
                          standalone_answer_format="original")
        return old_replace(value, **kwargs)

    def validate(result, entry):
        old_result(result, entry)
        selection, training = result["selection"], result["training"]
        final = result["standalone_cartridge"]
        memory = final["memory"]
        if (selection.get("selection_policy") != "decision_proper_where_v1" or
            selection.get("question_overlap_policy") != "proper_where" or
            selection.get("entity_reserve_fraction") != .25 or
            training.get("value_learning_rate") != .0001 or training.get("gain_learning_rate") != .01 or
            training.get("gain_head_mode") != "kv" or training.get("adaptive_max_answer_steps") != 8 or
            len(training["steps"]) != 2 + result["self_study"]["verified"] or
            training.get("student_question_matches_teacher") or final.get("answer_format") != "original" or
            result.get("output_source") != "standalone_cartridge" or
            not memory.get("kv_only_materialized") or memory.get("cartridge_adapter_bytes") != 0 or
            memory.get("additional_source_kv_bytes_before_decode") != 0 or
            memory.get("additional_training_adapter_bytes_before_decode") != 0):
            raise ValueError("The question-overlap independent cartridge contract failed")

    implementation.ROOT, implementation.validate_manifest = root, validate_manifest
    implementation.validate, implementation.replace = validate, configured
    runtime.FrozenJSONSelector, controls.original_question_controls = selector, physical_controls
    try:
        yield
    finally:
        (implementation.ROOT, implementation.validate_manifest, implementation.validate,
         implementation.replace, runtime.FrozenJSONSelector, controls.original_question_controls) = old
