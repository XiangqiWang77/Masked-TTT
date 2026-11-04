"""Run the two study-prompt arms using the existing single-GPU case driver."""
from contextlib import contextmanager, nullcontext

from experiments.prompt_alignment.same_gpu_pair import run_study_prompt_pair
from masked_ttt.config import TTTConfig


@contextmanager
def study_pair_runner(implementation, runtime, controls, *, root, selector, physical_controls):
    """Install one process-local paired adapter and restore it on every exit."""
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
           runtime.MaskedTTT, controls.original_question_controls)
    _, old_manifest, old_result, old_replace, old_preflight, _, old_method, old_controls = old

    def validate_manifest(manifest):
        if (manifest.get("mode") not in ("development", "holdout") or
            manifest.get("models") != ["qwen"] or
            manifest.get("selection_policy") != "decision_entity_link_v1" or
            manifest.get("entity_reserve_fraction") != .25 or
            manifest.get("student_rollout_tokens") != 8 or
            manifest.get("self_study_prompt_comparison") != "original_vs_study_only" or
            manifest.get("holdout_evaluated_at_selection") is not False):
            raise ValueError("The paired study-prompt protocol changed")
        return old_manifest(dict(manifest, selection_policy="cached_decision_score10"))

    def configured(value, **kwargs):
        if isinstance(value, TTTConfig) and "max_context" in kwargs:
            kwargs.update(steps=2, max_answer_steps=8, gain_head_mode="kv",
                          value_learning_rate=.0001, gain_learning_rate=.01,
                          align_student_teacher_prompt=False, standalone_answer_format="original")
        return old_replace(value, **kwargs)

    class StudyPair:
        def __init__(self, engine, selector_instance):
            self.engine, self.selector = engine, selector_instance

        def answer(self, request, *, time_budget_seconds=None):
            paired = run_study_prompt_pair(self.engine, self.selector, request, self.engine.config,
                                          time_budget_seconds=time_budget_seconds)
            baseline, candidate = paired["left"], paired["right"]
            candidate["study_prompt_pair"] = dict(
                baseline_prediction=baseline["prediction"],
                baseline_tokens=baseline["generated_token_ids"],
                baseline_verified=baseline["self_study"]["verified"],
                candidate_verified=candidate["self_study"]["verified"],
                baseline_training_steps=len(baseline["training"]["steps"]),
                candidate_training_steps=len(candidate["training"]["steps"]),
                same_practice_checks=paired["same_practice_checks"],
                same_loaded_model=paired["same_loaded_model"], same_gpu=paired["same_gpu"],
                same_request=paired["same_request"],
                independently_prefilled_caches=paired["independently_prefilled_caches"],
                same_selection=True, same_teacher_tokens=True, same_zero_tokens=True,
                baseline_standalone_released=True, candidate_standalone_released=True,
                baseline_native_prediction=baseline["self_judge"]["native_prediction"],
                reference_answers_used=False, teacher_probability_tensors_compared=False)
            return candidate

    def validate(result, entry):
        old_result(result, entry)
        p, t, selection = result["study_prompt_pair"], result["training"], result["selection"]
        if (selection.get("selection_policy") != "decision_entity_link_v1" or
            selection.get("entity_reserve_fraction") != .25 or
            t.get("gain_head_mode") != "kv" or t.get("gain_learning_rate") != .01 or
            t.get("value_learning_rate") != .0001 or t.get("student_question_matches_teacher") or
            t.get("adaptive_max_answer_steps") != 8 or
            len(t["steps"]) != 2 + result["self_study"]["verified"] or
            p["candidate_training_steps"] != 2 + p["candidate_verified"] or
            p["baseline_training_steps"] != 2 + p["baseline_verified"] or
            p["reference_answers_used"] or p["teacher_probability_tensors_compared"] or
            not all(p[key] for key in ("same_loaded_model", "same_gpu", "same_request",
                "independently_prefilled_caches", "same_selection", "same_teacher_tokens",
                "same_zero_tokens", "baseline_standalone_released", "candidate_standalone_released"))):
            raise ValueError("The two physically released SD arms did not match their protocol")

    def preflight(engine):
        # The ordinary upstream real-model preflight has self-study disabled;
        # do not route that synthetic request through the two-arm comparison.
        runtime.MaskedTTT, controls.original_question_controls = old_method, physical_controls
        try:
            return old_preflight(engine)
        finally:
            runtime.MaskedTTT, controls.original_question_controls = StudyPair, nullcontext

    implementation.ROOT, implementation.validate_manifest = root, validate_manifest
    implementation.validate, implementation.replace, implementation.preflight = validate, configured, preflight
    runtime.FrozenJSONSelector, runtime.MaskedTTT = selector, StudyPair
    controls.original_question_controls = nullcontext
    try:
        yield
    finally:
        (implementation.ROOT, implementation.validate_manifest, implementation.validate,
         implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
         runtime.MaskedTTT, controls.original_question_controls) = old
