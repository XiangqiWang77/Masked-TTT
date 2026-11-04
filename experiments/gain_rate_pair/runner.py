"""Compare attention-gain learning rates without changing the OLMo SD method."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace

from experiments.gain_rate_pair.protocol import BASE_GAIN, CANDIDATE_GAIN, VALUE_RATES
from experiments.prompt_alignment.same_gpu_pair import run_pair
from masked_ttt.config import TTTConfig

def model_name(engine):
    name = getattr(engine.model.config, "model_type", None)
    if name == "qwen3":
        return "qwen"
    if name == "gpt_oss":
        return "gpt"
    raise ValueError("The gain-rate protocol supports Qwen3 and GPT-OSS")


def configs(config, model):
    """Hold the model-specific Value rate fixed; change only the gain rate."""
    if model not in VALUE_RATES:
        raise ValueError("Unknown model for gain-rate comparison")
    common = replace(config, student_rollout_tokens=8, value_learning_rate=VALUE_RATES[model])
    return (replace(common, gain_learning_rate=BASE_GAIN),
            replace(common, gain_learning_rate=CANDIDATE_GAIN))


def real_pair_preflight(engine):
    """Verify both training and release paths on the allocated real model."""
    if engine.device.type != "cuda":
        raise ValueError("The real-model pair preflight requires an allocated GPU")
    from masked_ttt import Request
    from masked_ttt.request import prepare_layout
    from masked_ttt.selection import PreparedRequest

    request = Request(document="The old note is red. " + "Ordinary material. " * 30 +
                      "The current note is blue.", question="What is the current note?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            selected = [len(layout.chunks) // 2]
            return PreparedRequest(item, layout, selected, dict(backend="synthetic",
                selected_original_chunks=selected, selected_chunks=selected, total_chunks=len(layout.chunks),
                selected_tokens=sum(len(layout.chunks[i]) for i in selected)))

    old = engine.config
    try:
        common = replace(old, chunk_tokens=64, steps=2, max_answer_steps=2, self_question_count=0,
            guard_enabled=False, gain_head_mode="kv", selected_transfer_weight=0.,
            value_basis_source="document", standalone_answer_format="original",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        left, right = configs(common, model_name(engine))
        pair = run_pair(engine, Selector(), request, left, right, time_budget_seconds=300)
        if (not pair["same_gpu"] or not pair["same_loaded_model"] or not pair["same_request"] or
            not pair["independently_prefilled_caches"] or pair["reference_answers_used"] or
            pair["changed_settings"] != ["gain_learning_rate"]):
            raise AssertionError("The real-GPU gain-rate pair violated its contract")
        return dict(paired_gain_rate_preflight=True, same_gpu=True, same_teacher_and_zero=True,
                    independently_materialized_and_released=True, reference_answers_used=False)
    finally:
        engine.config = old


@contextmanager
def pair_runner(implementation, runtime, controls, *, root, selector, physical_controls):
    """Install the controlled pair in a single-request frozen benchmark process."""
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
           runtime.MaskedTTT, controls.original_question_controls)
    _, old_manifest, old_result, old_replace, old_preflight, _, old_method, _ = old

    def validate_manifest(manifest):
        if (manifest.get("mode") not in ("development", "holdout") or
            not manifest.get("models") or any(x not in VALUE_RATES for x in manifest["models"]) or
            manifest.get("selection_policy") != "decision_entity_link_v1" or
            manifest.get("entity_reserve_fraction") != .25 or manifest.get("student_rollout_tokens") != 8 or
            manifest.get("baseline_gain_lr") != BASE_GAIN or
            manifest.get("candidate_gain_lr") != CANDIDATE_GAIN or
            manifest.get("value_learning_rates") != VALUE_RATES or
            manifest.get("holdout_evaluated_at_selection") is not False):
            raise ValueError("The paired gain-rate protocol changed")
        return old_manifest(dict(manifest, selection_policy="cached_decision_score10"))

    def configured(value, **kwargs):
        if isinstance(value, TTTConfig) and "max_context" in kwargs:
            kwargs.update(steps=2, max_answer_steps=8, gain_head_mode="kv", value_learning_rate=.0001,
                          gain_learning_rate=BASE_GAIN, align_student_teacher_prompt=False,
                          standalone_answer_format="original")
        return old_replace(value, **kwargs)

    class PairedMethod:
        def __init__(self, engine, selector_instance):
            self.engine, self.selector = engine, selector_instance

        def answer(self, request, *, time_budget_seconds=None):
            model = model_name(self.engine)
            baseline_config, candidate_config = configs(self.engine.config, model)
            paired = run_pair(self.engine, self.selector, request, baseline_config, candidate_config,
                              time_budget_seconds=time_budget_seconds)
            baseline, candidate = paired["left"], paired["right"]
            candidate["gain_rate_pair"] = dict(model=model, baseline_prediction=baseline["prediction"],
                baseline_tokens=baseline["generated_token_ids"],
                baseline_gain_lr=baseline["training"]["gain_learning_rate"],
                candidate_gain_lr=candidate["training"]["gain_learning_rate"],
                value_lr=baseline["training"]["value_learning_rate"],
                baseline_verified=baseline["self_study"]["verified"],
                candidate_verified=candidate["self_study"]["verified"],
                baseline_training_steps=len(baseline["training"]["steps"]),
                candidate_training_steps=len(candidate["training"]["steps"]),
                same_practice_checks=paired["same_practice_checks"],
                same_loaded_model=paired["same_loaded_model"], same_gpu=paired["same_gpu"],
                same_request=paired["same_request"], independently_prefilled_caches=paired["independently_prefilled_caches"],
                same_selection=True, same_teacher_tokens=True, same_zero_tokens=True,
                baseline_standalone_released=True, candidate_standalone_released=True,
                baseline_native_prediction=baseline["self_judge"]["native_prediction"],
                changed_settings=paired["changed_settings"], reference_answers_used=False,
                teacher_probability_tensors_compared=False)
            return candidate

    def validate(result, entry):
        old_result(result, entry)
        p, t, selection = result["gain_rate_pair"], result["training"], result["selection"]
        if (selection.get("selection_policy") != "decision_entity_link_v1" or
            selection.get("entity_reserve_fraction") != .25 or t.get("gain_head_mode") != "kv" or
            p["model"] not in VALUE_RATES or t.get("gain_learning_rate") != CANDIDATE_GAIN or
            t.get("value_learning_rate") != VALUE_RATES[p["model"]] or p["value_lr"] != VALUE_RATES[p["model"]] or
            p["baseline_gain_lr"] != BASE_GAIN or p["candidate_gain_lr"] != CANDIDATE_GAIN or
            t.get("student_rollout_limit") != 8 or t.get("student_question_matches_teacher") or
            t.get("adaptive_max_answer_steps") != 8 or
            len(t["steps"]) != 2 + result["self_study"]["verified"] or
            p["candidate_training_steps"] != 2 + p["candidate_verified"] or
            p["baseline_training_steps"] != 2 + p["baseline_verified"] or
            p["changed_settings"] != ["gain_learning_rate"] or
            p["reference_answers_used"] or p["teacher_probability_tensors_compared"] or
            not all(p[key] for key in ("same_loaded_model", "same_gpu", "same_request",
                "independently_prefilled_caches", "same_selection", "same_teacher_tokens", "same_zero_tokens",
                "same_practice_checks", "baseline_standalone_released", "candidate_standalone_released"))):
            raise ValueError("The paired original-SD gain-rate arms changed")

    def preflight(engine):
        runtime.MaskedTTT, controls.original_question_controls = old_method, physical_controls
        try:
            baseline = old_preflight(engine)
        finally:
            runtime.MaskedTTT, controls.original_question_controls = PairedMethod, nullcontext
        return dict(baseline, **real_pair_preflight(engine))

    implementation.ROOT, implementation.validate_manifest = root, validate_manifest
    implementation.validate, implementation.replace, implementation.preflight = validate, configured, preflight
    runtime.FrozenJSONSelector, runtime.MaskedTTT = selector, PairedMethod
    controls.original_question_controls = nullcontext
    try:
        yield
    finally:
        (implementation.ROOT, implementation.validate_manifest, implementation.validate,
         implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
         runtime.MaskedTTT, controls.original_question_controls) = old
