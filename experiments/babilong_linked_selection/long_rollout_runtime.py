"""Compose the two OLMo-style settings without changing the SD objective.

This adapter is for the existing single-GPU BABILONG runner. It replaces
only its frozen-score selector and paired-decoding context, and verifies the
linked selection budget and physical release on every result. The underlying
runner still performs selected-teacher/full-student on-policy KL with the
same model, update parameters, and training procedure.
"""
from contextlib import contextmanager


@contextmanager
def linked_long_rollout_runner(implementation, runtime, paired_controls, *, root,
                               linked_selector, physical_controls):
    """Temporarily compose a single-request runner; restore it on failure."""
    original = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
                runtime.FrozenJSONSelector, paired_controls.original_question_controls)
    old_root, original_manifest, original_result, old_selector, old_controls = original

    def validate_manifest(manifest):
        if (manifest.get("selection_policy") != "decision_entity_link_v1" or
            manifest.get("entity_reserve_fraction") != .25 or
            manifest.get("student_rollout_tokens") != 32):
            raise ValueError("The linked selector or 32-token budget changed")
        # The upstream validator checks all other frozen OLMo settings. It
        # expects the score-only label, so supply that label to its copy only.
        upstream = dict(manifest, selection_policy="cached_decision_score10")
        return original_manifest(upstream)

    def validate_result(result, entry):
        original_result(result, entry)
        selection = result["selection"]
        memory = result["standalone_cartridge"]["memory"]
        if (selection.get("selection_policy") != "decision_entity_link_v1" or
            selection.get("entity_reserve_fraction") != .25 or
            result.get("output_source") != "standalone_cartridge" or
            result.get("prediction") != result["standalone_cartridge"].get("prediction") or
            result.get("generated_token_ids") != result["standalone_cartridge"].get("generated_token_ids") or
            not memory.get("kv_only_materialized") or memory.get("cartridge_adapter_bytes") != 0 or
            memory.get("additional_source_kv_bytes_before_decode") != 0 or
            memory.get("additional_training_adapter_bytes_before_decode") != 0):
            raise ValueError("Linked selection or independent KV-only release contract failed")

    implementation.ROOT = root
    implementation.validate_manifest = validate_manifest
    implementation.validate = validate_result
    runtime.FrozenJSONSelector = linked_selector
    paired_controls.original_question_controls = physical_controls
    try:
        yield
    finally:
        (implementation.ROOT, implementation.validate_manifest, implementation.validate,
         runtime.FrozenJSONSelector, paired_controls.original_question_controls) = original
