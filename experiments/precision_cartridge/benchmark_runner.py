"""Pair native and float32 materialization without changing OLMo-style SD."""
from contextlib import contextmanager
from dataclasses import replace

from experiments.babilong_linked_selection.long_rollout_runtime import linked_long_rollout_runner
from experiments.anchored_short_verifier.paired_controls import original_question_controls as _original_question_controls
from experiments.precision_cartridge.paired_diagnostic import capture_precision_paired
from masked_ttt.kv_only import kv_only_cartridge


@contextmanager
def precision_controls():
    # Both independently materialized cartridges drop duplicate adapter tensors.
    # Capture the unpatched control before the outer benchmark runner swaps
    # its public hook to this function; importing it here would recurse.
    with kv_only_cartridge(), capture_precision_paired(include_noop=True), _original_question_controls():
        yield


def preflight(engine):
    """Exercise paired materialization and physical release on the loaded model."""
    from masked_ttt import MaskedTTT, Request
    from masked_ttt.request import prepare_layout
    from masked_ttt.selection import PreparedRequest

    request = Request(document="The old note is red. " + "Ordinary material. " * 30 +
                      "The current note is blue.", question="What is the current note?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [len(layout.chunks) // 2], dict(backend="synthetic"))

    old = engine.config
    try:
        engine.config = replace(old, chunk_tokens=64, steps=2, max_answer_steps=2, self_question_count=0,
            student_rollout_tokens=3, guard_enabled=False, gain_head_mode="kv", selected_transfer_weight=0.,
            value_basis_source="document", release_full_before_cartridge=True,
            release_selected_source_before_decode=True, standalone_answer_format="original")
        with precision_controls():
            result = MaskedTTT(engine, Selector()).answer(request, time_budget_seconds=300)
        diag = result["standalone_cartridge"]["precision_paired_diagnostic"]
        physical = result["standalone_cartridge"]["memory"]
        steps = result["training"]["steps"]
        if (not diag["passed"] or not diag["precision_kv_only_materialized"] or
            diag["precision_adapter_bytes"] != 0 or
            diag["precision_tokens"] != diag["precision_matched_adapted_tokens"] or
            not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
            not any(item["gain_gradient_norm"] > 0 for item in steps) or
            not any(item["value_gradient_norm"] > 0 for item in steps) or
            result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0 or
            not physical["kv_only_materialized"] or physical["additional_source_kv_bytes_before_decode"] != 0 or
            physical["additional_training_adapter_bytes_before_decode"] != 0):
            raise AssertionError("The paired precision or independent cartridge contract failed")
        return dict(passed=True, same_training_state=True, both_gradient_paths=True,
                    independent_selected_sources=True, no_reference_answers=True)
    finally:
        engine.config = old


@contextmanager
def precision_runner(implementation, runtime, paired_controls, *, root, linked_selector):
    old_preflight = implementation.preflight
    with linked_long_rollout_runner(implementation, runtime, paired_controls, root=root,
                                    linked_selector=linked_selector, physical_controls=precision_controls):
        original_validate = implementation.validate

        def validate(result, entry):
            original_validate(result, entry)
            diag = result["standalone_cartridge"]["precision_paired_diagnostic"]
            if (result["training"]["gain_head_mode"] != "kv" or
                len(result["training"]["steps"]) != 2 + result["self_study"]["verified"] or not diag["passed"] or
                not diag["precision_kv_only_materialized"] or diag["precision_adapter_bytes"] != 0 or
                diag["precision_tokens"] != diag["precision_matched_adapted_tokens"] or
                diag["selected_source_bytes_before_decode"] != 0 or
                diag["training_bytes_before_decode"] != 0 or
                not diag["same_selected_kv"] or not diag["same_question_suffix"] or
                not diag["same_generation_limit"] or not diag["same_precision_backend"] or
                diag["reference_answers_used"] or diag["teacher_answer_in_student_prompt"]):
                raise ValueError("The paired precision or original SD contract failed")

        implementation.preflight = preflight
        implementation.validate = validate
        try:
            yield
        finally:
            implementation.preflight = old_preflight
            implementation.validate = original_validate
