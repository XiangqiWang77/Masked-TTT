"""Run the frozen linked-evidence SD path with a matched main question."""

from contextlib import contextmanager
from experiments.babilong_linked_selection.long_rollout_runtime import linked_long_rollout_runner
from masked_ttt.config import TTTConfig


@contextmanager
def aligned_prompt_runner(implementation, runtime, paired_controls, *, root,
                          linked_selector, physical_controls):
    old_replace, old_preflight = implementation.replace, implementation.preflight
    with linked_long_rollout_runner(implementation, runtime, paired_controls, root=root,
                                    linked_selector=linked_selector, physical_controls=physical_controls):
        old_validate = implementation.validate

        def configured_replace(value, **kwargs):
            if isinstance(value, TTTConfig) and "max_context" in kwargs and "steps" not in kwargs:
                kwargs.update(steps=2, max_answer_steps=2, gain_head_mode="kv",
                              align_student_teacher_prompt=True,
                              standalone_answer_format="original")
            return old_replace(value, **kwargs)

        def validate(result, entry):
            old_validate(result, entry)
            training = result["training"]
            if (not training.get("student_question_matches_teacher") or
                training.get("gain_head_mode") != "kv" or
                len(training["steps"]) != 2 + result["self_study"]["verified"] or
                training.get("selected_transfer_weight") != 0 or
                result["standalone_cartridge"].get("answer_format") != "original"):
                raise ValueError("Aligned main SD question or original cartridge format changed")

        def preflight(engine):
            if not engine.config.align_student_teacher_prompt:
                raise ValueError("The real-model preflight must use aligned questions")
            result = old_preflight(engine)
            return dict(result, aligned_main_question=True)

        implementation.replace = configured_replace
        implementation.validate = validate
        implementation.preflight = preflight
        try:
            yield
        finally:
            implementation.replace = old_replace
            implementation.validate = old_validate
            implementation.preflight = old_preflight
