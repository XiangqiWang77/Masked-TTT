"""Compare two original-method SD settings on one request and one GPU.

Each arm independently rebuilds the full and selected caches and physically
releases them after materialization. The teacher and zero controls are checked
for equality; this does not claim their full probability tensors are identical.
This diagnostic uses a process-local control hook and must run without
concurrent requests in the same process.
"""
from contextlib import nullcontext
from copy import copy
from dataclasses import fields

import torch

from experiments.anchored_short_verifier.paired_controls import original_question_controls
from masked_ttt import MaskedTTT


ALLOWED_DIFFERENCES = frozenset((
    "gain_learning_rate", "value_learning_rate", "steps", "max_answer_steps",
    "answer_KL_stop", "student_rollout_tokens", "align_student_teacher_prompt",
))
SELECTION_FIELDS = ("selected_original_chunks", "selected_chunks", "total_chunks", "selected_tokens")


def validate_configs(left, right):
    """Hold all inference settings fixed except explicit SD optimizer choices."""
    changed = {field.name for field in fields(left)
               if getattr(left, field.name) != getattr(right, field.name)}
    if not changed.issubset(ALLOWED_DIFFERENCES):
        raise ValueError("Only the specified SD optimization settings may differ")
    for config in (left, right):
        if (config.selected_transfer_weight != 0 or config.value_basis_source != "document" or
            config.guard_enabled or config.standalone_answer_format != "original" or
            not config.standalone_cartridge_enabled or not config.release_full_before_cartridge or
            not config.release_selected_source_before_decode):
            raise ValueError("The original standalone on-policy SD method is required")
    return sorted(changed)


def validate_pair(left, right, *, require_same_practices=True):
    if type(require_same_practices) is not bool:
        raise ValueError("Practice comparison must be explicitly enabled or disabled")
    for result in (left, right):
        if (not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
            result["teacher_answer_in_final_prompt"] or
            result["self_study"]["reference_answers_used"] or
            result["self_study"]["student_answer_hint_used"] or
            result["training"]["objective"] != "on-policy KL(teacher || student)" or
            result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0):
            raise ValueError("A paired result violates the original-method contract")
        memory = result["standalone_cartridge"]["memory"]
        control = result["standalone_cartridge"]["original_question_controls"]
        if (not memory["kv_only_materialized"] or memory["additional_source_kv_bytes_before_decode"] != 0 or
            memory["additional_training_adapter_bytes_before_decode"] != 0 or
            not control["same_selected_kv"] or not control["same_question_suffix"] or
            not control["same_generation_limit"] or control["teacher_answer_in_student_prompt"] or
            control["reference_answers_used"]):
            raise ValueError("The independent cartridge or zero-update control is invalid")
    if left["request_chat_date"] != right["request_chat_date"]:
        raise ValueError("The two request dates differ")
    if any(left["selection"][key] != right["selection"][key] for key in SELECTION_FIELDS):
        raise ValueError("The two selections differ")
    if left["teacher"]["tokens"] != right["teacher"]["tokens"]:
        raise ValueError("The two Teacher output sequences differ")
    same_practices = left["self_study"]["checks"] == right["self_study"]["checks"]
    if require_same_practices and not same_practices:
        raise ValueError("The two verified practice sets differ")
    zero = lambda result: result["standalone_cartridge"]["original_question_controls"]["zero"]["tokens"]
    if zero(left) != zero(right):
        raise ValueError("The two same-format zero-update sequences differ")
    return same_practices


def _run_pair(engine, selector, request, left_config, right_config, *, contexts,
              require_same_practices, time_budget_seconds):
    changed = validate_configs(left_config, right_config)
    devices = {parameter.device for parameter in engine.model.parameters()}
    if devices != {engine.device}:
        raise ValueError("This strict paired diagnostic requires a single-device model")
    stamp = engine.weight_stamp()
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state(engine.device) if engine.device.type == "cuda" else None
    scope = getattr(engine.tokenizer, "request_scope", None)
    results = []
    try:
        with (scope() if callable(scope) else nullcontext()), original_question_controls():
            for config, context in zip((left_config, right_config), contexts):
                torch.set_rng_state(cpu_rng)
                if gpu_rng is not None:
                    torch.cuda.set_rng_state(gpu_rng, engine.device)
                local = copy(engine)
                local.config = config
                with context():
                    results.append(MaskedTTT(local, selector).answer(request,
                        time_budget_seconds=time_budget_seconds, compact_standalone=True))
        engine.assert_frozen(stamp)
        same_practices = validate_pair(*results, require_same_practices=require_same_practices)
    finally:
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, engine.device)
    return dict(left=results[0], right=results[1], changed_settings=changed,
                same_loaded_model=True, same_device=True, same_gpu=engine.device.type == "cuda",
                device_type=engine.device.type, same_request=True,
                independently_prefilled_caches=True, reference_answers_used=False,
                teacher_probability_tensors_compared=False, same_practice_checks=same_practices)


def run_pair(engine, selector, request, left_config, right_config, *, time_budget_seconds=None):
    """Run two strict SD controls sequentially on the same loaded frozen model.

    The two cache prefills are independent. This returns raw results to the
    caller for post-generation scoring and never reads reference answers.
    """
    return _run_pair(engine, selector, request, left_config, right_config,
                     contexts=(nullcontext, nullcontext), require_same_practices=True,
                     time_budget_seconds=time_budget_seconds)


def run_study_prompt_pair(engine, selector, request, config, *, time_budget_seconds=None):
    """Compare the existing and the study-only question prompt, without labels.

    Different practices are expected, but Teacher output, selection and the
    zero-update answer still have to agree. Both arms use the same SD settings.
    """
    if config.self_question_count < 1:
        raise ValueError("The study-prompt comparison needs self-study enabled")
    from experiments.self_study_prompt.study_context import study_only_prompt
    result = _run_pair(engine, selector, request, config, config,
        contexts=(nullcontext, study_only_prompt), require_same_practices=False,
        time_budget_seconds=time_budget_seconds)
    result.update(changed_settings=["self_study_prompt"],
                  left_study_prompt="original", right_study_prompt="study_only")
    return result
