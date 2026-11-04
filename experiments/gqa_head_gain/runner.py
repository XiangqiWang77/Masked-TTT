"""Use per-query-head gains with the otherwise unchanged OLMo-style runner."""
from contextlib import contextmanager

from masked_ttt.config import TTTConfig


@contextmanager
def query_head_runner(implementation, runtime, paired_controls, *, root,
                      linked_selector, physical_controls):
    original = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
                implementation.replace, runtime.FrozenJSONSelector,
                paired_controls.original_question_controls)
    _, original_manifest, original_result, original_replace, _, _ = original

    def validate_manifest(manifest):
        if (manifest.get("gain_head_mode") != "query" or
            manifest.get("selection_policy") != "decision_entity_link_v1" or
            manifest.get("entity_reserve_fraction") != .25 or
            manifest.get("student_rollout_tokens") != 8):
            raise ValueError("The query-head comparison protocol changed")
        return original_manifest(dict(manifest, selection_policy="cached_decision_score10"))

    def validate_result(result, entry):
        original_result(result, entry)
        selection, training = result["selection"], result["training"]
        final = result["standalone_cartridge"]
        memory = final["memory"]
        if (training.get("gain_head_mode") != "query" or
            selection.get("selection_policy") != "decision_entity_link_v1" or
            selection.get("entity_reserve_fraction") != .25 or
            result.get("output_source") != "standalone_cartridge" or
            result.get("prediction") != final.get("prediction") or
            result.get("generated_token_ids") != final.get("generated_token_ids") or
            not memory.get("kv_only_materialized") or memory.get("cartridge_adapter_bytes") != 0 or
            memory.get("additional_source_kv_bytes_before_decode") != 0 or
            memory.get("additional_training_adapter_bytes_before_decode") != 0):
            raise ValueError("Query-head or physically independent cartridge contract failed")

    def configured_replace(value, **kwargs):
        if isinstance(value, TTTConfig):
            kwargs["gain_head_mode"] = "query"
        return original_replace(value, **kwargs)

    implementation.ROOT = root
    implementation.validate_manifest = validate_manifest
    implementation.validate = validate_result
    implementation.replace = configured_replace
    runtime.FrozenJSONSelector = linked_selector
    paired_controls.original_question_controls = physical_controls
    try:
        yield
    finally:
        (implementation.ROOT, implementation.validate_manifest, implementation.validate,
         implementation.replace, runtime.FrozenJSONSelector,
         paired_controls.original_question_controls) = original
