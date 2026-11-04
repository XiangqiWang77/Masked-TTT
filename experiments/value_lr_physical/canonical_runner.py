"""Run the paired tokenization comparison in the established GPU case driver."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace

from experiments.value_lr_physical.canonical_pair import run_pair
from masked_ttt.config import TTTConfig


def real_pair_preflight(engine):
    """Exercise both independent cartridge paths on the allocated real model."""
    if engine.device.type != "cuda":
        raise ValueError("The real-model preflight requires an allocated GPU")
    from masked_ttt import Request
    from masked_ttt.request import prepare_layout
    from masked_ttt.selection import PreparedRequest

    request = Request(document="The old note is red. " + "Ordinary material. " * 30 +
                      "The current note is blue.", question="What is the current note?", max_new_tokens=3)

    class SyntheticSelector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            selected = [len(layout.chunks) // 2]
            metadata = dict(backend="synthetic", selected_original_chunks=selected,
                selected_chunks=selected, total_chunks=len(layout.chunks),
                selected_tokens=sum(len(layout.chunks[index]) for index in selected))
            return PreparedRequest(item, layout, selected, metadata)

    old = engine.config
    try:
        engine.config = replace(old, chunk_tokens=64, steps=2, max_answer_steps=2,
            self_question_count=0, student_rollout_tokens=3, guard_enabled=True,
            selected_transfer_weight=0., value_basis_source="document", gain_head_mode="kv",
            standalone_answer_format="original", release_full_before_cartridge=True,
            release_selected_source_before_decode=True)
        pair = run_pair(engine, SyntheticSelector(), request, engine.config, time_budget_seconds=300)
        if (not pair["same_gpu"] or not pair["same_loaded_model"] or not pair["same_request"] or
            not pair["independently_prefilled_caches"] or pair["reference_answers_used"]):
            raise AssertionError("Real-model canonical comparison violated its paired contract")
        for arm in (pair["left"], pair["right"]):
            steps = arm["training"]["steps"]
            if not any(step["gain_gradient_norm"] > 0 for step in steps) or not any(
                step["value_gradient_norm"] > 0 for step in steps):
                raise AssertionError("Real-model canonical arm lost a gradient path")
        return dict(canonical_paired_preflight=True, same_gpu=True,
                    independently_materialized_and_released=True, reference_answers_used=False)
    finally:
        engine.config = old


@contextmanager
def canonical_pair_runner(implementation, runtime, controls, *, root, selector, physical_controls):
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
           runtime.MaskedTTT, controls.original_question_controls)
    _, old_manifest, old_result, old_replace, old_preflight, _, old_method, _ = old

    def validate_manifest(manifest):
        if (manifest.get("mode") not in ("development", "holdout") or
            not manifest.get("models") or any(name not in ("qwen", "gpt") for name in manifest["models"]) or
            manifest.get("selection_policy") != "decision_entity_link_v1" or
            manifest.get("entity_reserve_fraction") != .25 or
            manifest.get("student_rollout_tokens") != 8 or
            manifest.get("evidence_tokenization_comparison") != "original_vs_canonical_fragment" or
            manifest.get("holdout_evaluated_at_selection") is not False):
            raise ValueError("The paired canonical-evidence protocol changed")
        return old_manifest(dict(manifest, selection_policy="cached_decision_score10"))

    def configured(value, **kwargs):
        if isinstance(value, TTTConfig) and "max_context" in kwargs:
            kwargs.update(steps=2, max_answer_steps=8, gain_head_mode="kv",
                          value_learning_rate=.0001, gain_learning_rate=.01,
                          align_student_teacher_prompt=False, standalone_answer_format="original")
        return old_replace(value, **kwargs)

    class CanonicalPair:
        def __init__(self, engine, selector_instance):
            self.engine, self.selector = engine, selector_instance

        def answer(self, request, *, time_budget_seconds=None):
            paired = run_pair(self.engine, self.selector, request, self.engine.config,
                              time_budget_seconds=time_budget_seconds)
            baseline, candidate = paired["left"], paired["right"]
            control = baseline["standalone_cartridge"]["original_question_controls"]
            candidate["canonical_tokenization_pair"] = dict(
                baseline_prediction=baseline["prediction"], baseline_tokens=baseline["generated_token_ids"],
                baseline_zero_prediction=control["zero"]["prediction"], baseline_zero_tokens=control["zero"]["tokens"],
                baseline_teacher_prediction=baseline["teacher"]["prediction"],
                baseline_teacher_tokens=baseline["teacher"]["tokens"],
                baseline_teacher_prompt_tokens=baseline["teacher"]["prompt_tokens"],
                candidate_teacher_prompt_tokens=candidate["teacher"]["prompt_tokens"],
                baseline_native_prediction=baseline["self_judge"]["native_prediction"],
                baseline_verified=baseline["self_study"]["verified"],
                candidate_verified=candidate["self_study"]["verified"],
                same_practice_checks=paired["same_practice_checks"],
                same_teacher_tokens=paired["same_teacher_tokens"], same_zero_tokens=paired["same_zero_tokens"],
                same_loaded_model=paired["same_loaded_model"], same_gpu=paired["same_gpu"],
                same_request=paired["same_request"], independently_prefilled_caches=paired["independently_prefilled_caches"],
                same_selection=True, baseline_standalone_released=True, candidate_standalone_released=True,
                reference_answers_used=False, teacher_probability_tensors_compared=False)
            return candidate

    def validate(result, entry):
        old_result(result, entry)
        p, t, selection = result["canonical_tokenization_pair"], result["training"], result["selection"]
        if (selection.get("selection_policy") != "decision_entity_link_v1" or
            selection.get("entity_reserve_fraction") != .25 or t.get("gain_head_mode") != "kv" or
            t.get("gain_learning_rate") != .01 or t.get("value_learning_rate") != .0001 or
            t.get("student_question_matches_teacher") or t.get("adaptive_max_answer_steps") != 8 or
            len(t["steps"]) != 2 + result["self_study"]["verified"] or
            p["candidate_verified"] != result["self_study"]["verified"] or
            p["baseline_native_prediction"] != result["self_judge"]["native_prediction"] or
            p["reference_answers_used"] or p["teacher_probability_tensors_compared"] or
            not all(p[key] for key in ("same_loaded_model", "same_gpu", "same_request",
                "independently_prefilled_caches", "same_selection", "baseline_standalone_released",
                "candidate_standalone_released"))):
            raise ValueError("The physically released canonical SD arms did not match their protocol")

    def preflight(engine):
        runtime.MaskedTTT, controls.original_question_controls = old_method, physical_controls
        try:
            baseline = old_preflight(engine)
        finally:
            runtime.MaskedTTT, controls.original_question_controls = CanonicalPair, nullcontext
        return dict(baseline, **real_pair_preflight(engine))

    implementation.ROOT, implementation.validate_manifest = root, validate_manifest
    implementation.validate, implementation.replace, implementation.preflight = validate, configured, preflight
    runtime.FrozenJSONSelector, runtime.MaskedTTT = selector, CanonicalPair
    controls.original_question_controls = nullcontext
    try:
        yield
    finally:
        (implementation.ROOT, implementation.validate_manifest, implementation.validate,
         implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
         runtime.MaskedTTT, controls.original_question_controls) = old
