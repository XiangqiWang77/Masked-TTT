"""Compose the original 32-token SD runner with 2/4/8-step diagnostics."""
from contextlib import contextmanager
from dataclasses import replace

from experiments.babilong_linked_selection.long_rollout_runtime import linked_long_rollout_runner
from experiments.long_rollout_steps.probe import capture_step_snapshots
from masked_ttt.config import TTTConfig


@contextmanager
def checkpoint_controls(physical_controls, *, final_base_step=8):
    with physical_controls(), capture_step_snapshots(final_base_step=final_base_step):
        yield


def preflight(engine, *, final_base_step=8):
    """Run the actual model and verify the fixed checkpoint and final release."""
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from masked_ttt import MaskedTTT, Request
    from masked_ttt.request import prepare_layout
    from masked_ttt.selection import PreparedRequest

    request = Request(document="The old note is red. " + "Ordinary material. " * 30 +
                      "The current note is blue.", question="What is the current note?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [len(layout.chunks) // 2], dict(backend="synthetic"))

    if final_base_step not in (2, 4, 8):
        raise ValueError("The fixed preflight step must be 2, 4, or 8")
    baseline_step = 2 if final_base_step == 8 else final_base_step
    old = engine.config
    try:
        engine.config = replace(old, chunk_tokens=64, steps=baseline_step,
            max_answer_steps=baseline_step, self_question_count=0,
            student_rollout_tokens=3, guard_enabled=False, gain_head_mode="kv", selected_transfer_weight=0.,
            value_basis_source="document", release_full_before_cartridge=True,
            release_selected_source_before_decode=True, standalone_answer_format="original")
        method = MaskedTTT(engine, Selector())
        with kv_only_original_question_controls():
            baseline = method.answer(request, time_budget_seconds=300)
        engine.config = replace(engine.config, steps=8, max_answer_steps=8)
        with checkpoint_controls(kv_only_original_question_controls, final_base_step=final_base_step):
            result = method.answer(request, time_budget_seconds=300)
        steps = result["training"]["steps"]
        checkpoints = result["standalone_cartridge"]["step_probe"]["checkpoints"]
        physical = result["standalone_cartridge"]["memory"]
        if (len(steps) != 8 or checkpoints[str(baseline_step)]["tokens"] != baseline["generated_token_ids"] or
            checkpoints[str(final_base_step)]["tokens"] != result["generated_token_ids"] or
            result["standalone_cartridge"]["step_probe"]["fixed_final_base_step"] != final_base_step or
            not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
            not any(item["gain_gradient_norm"] > 0 for item in steps) or
            not any(item["value_gradient_norm"] > 0 for item in steps) or
            result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0 or
            not physical["kv_only_materialized"] or physical["additional_source_kv_bytes_before_decode"] != 0 or
            physical["additional_training_adapter_bytes_before_decode"] != 0):
            raise AssertionError("The original SD checkpoint or standalone release contract failed")
        return dict(passed=True, baseline_base_step=baseline_step, fixed_final_base_step=final_base_step,
                    same_baseline_checkpoint_tokens=True, same_fixed_final_tokens=True,
                    same_two_step_tokens=baseline_step == 2,
                    same_eight_step_tokens=final_base_step == 8,
                    both_gradient_paths=True, independent_cartridge=True)
    finally:
        engine.config = old


@contextmanager
def long_steps_runner(implementation, runtime, paired_controls, *, root, linked_selector, physical_controls):
    original_replace, original_preflight = implementation.replace, implementation.preflight
    with linked_long_rollout_runner(implementation, runtime, paired_controls, root=root,
                                    linked_selector=linked_selector, physical_controls=physical_controls):
        original_validate = implementation.validate

        def configured_replace(value, **kwargs):
            if isinstance(value, TTTConfig) and "max_context" in kwargs and "steps" not in kwargs:
                kwargs.update(steps=8, max_answer_steps=8, gain_head_mode="kv")
            return original_replace(value, **kwargs)

        def validate(result, entry):
            original_validate(result, entry)
            training = result["training"]
            probe = result["standalone_cartridge"]["step_probe"]
            practices = result["self_study"]["verified"]
            if (training["gain_head_mode"] != "kv" or len(training["steps"]) != 8 + practices or
                set(probe["checkpoints"]) != {"2", "4", "8"} or probe["practice_count"] != practices or
                any(probe["checkpoints"][str(step)]["total_steps"] != step + practices for step in (2, 4, 8)) or
                probe["checkpoints"]["8"]["tokens"] != result["generated_token_ids"] or
                not probe["same_selected_kv"] or probe["full_kv_used_at_decode"] or
                probe["reference_answers_used"] or probe["teacher_answer_in_final_prompt"] or
                not probe["diagnostic_snapshots_offloaded_to_cpu"] or
                not probe["diagnostic_snapshots_released_before_final_decode"]):
                raise ValueError("The 2/4/8-step original-SD checkpoint contract failed")

        implementation.replace = configured_replace
        implementation.preflight = preflight
        implementation.validate = validate
        paired_controls.original_question_controls = lambda: checkpoint_controls(physical_controls)
        try:
            yield
        finally:
            implementation.replace = original_replace
            implementation.preflight = original_preflight
            implementation.validate = original_validate
