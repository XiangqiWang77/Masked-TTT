"""Run a fixed 8-step original-SD cartridge with physical KV-only decoding."""
from contextlib import contextmanager

from experiments.long_rollout_steps.runner import checkpoint_controls, preflight as checkpoint_preflight
from experiments.value_lr_physical.runner import physical_value_runner


@contextmanager
def physical_step_runner(implementation, paired_controls, *, root, fixed_steps, physical_controls):
    if fixed_steps != 8:
        raise ValueError("This frozen checkpoint runner requires the selected 8-step candidate")
    controls = lambda: checkpoint_controls(physical_controls, final_base_step=fixed_steps)
    with physical_value_runner(implementation, paired_controls, root=root, fixed_lr=.0001,
                               physical_controls=controls, base_steps=fixed_steps):
        old_preflight, old_validate = implementation.preflight, implementation.validate

        def preflight(engine):
            return checkpoint_preflight(engine, final_base_step=fixed_steps)

        def validate(result, entry):
            old_validate(result, entry)
            final = result["standalone_cartridge"]
            probe = final["step_probe"]
            practices = result["self_study"]["verified"]
            if (probe["fixed_final_base_step"] != fixed_steps or probe["practice_count"] != practices or
                set(probe["checkpoints"]) != {"2", "4", "8"} or
                any(probe["checkpoints"][str(step)]["total_steps"] != step + practices for step in (2, 4, 8)) or
                probe["checkpoints"]["8"]["tokens"] != final["generated_token_ids"] or
                not probe["same_selected_kv"] or probe["full_kv_used_at_decode"] or
                probe["reference_answers_used"] or probe["teacher_answer_in_final_prompt"] or
                not probe["diagnostic_snapshots_offloaded_to_cpu"] or
                not probe["diagnostic_snapshots_released_before_final_decode"]):
                raise ValueError("The fixed-step or independent-cartridge checkpoint contract failed")
            result["training"]["fixed_base_steps"] = fixed_steps

        implementation.preflight = preflight
        implementation.validate = validate
        try:
            yield
        finally:
            implementation.preflight = old_preflight
            implementation.validate = old_validate
