"""Same-device comparison of two selected-evidence tokenization plans.

Both arms use the original full-context Student, selected Teacher, on-policy KL,
gain and value updates, and independently materialized selected KV. Only the
selected document fragment's tokenization differs. This is a process-local
diagnostic: it must not run concurrently with other requests in the process.
"""
from contextlib import nullcontext
from copy import copy
import time

import torch

from experiments.anchored_short_verifier.paired_controls import original_question_controls
from experiments.prompt_alignment.same_gpu_pair import SELECTION_FIELDS, validate_configs
from experiments.value_lr_physical.canonical_context import canonical_selected_evidence
from masked_ttt import MaskedTTT


def validate_pair(left, right):
    """Check matching sources and actual cache release, allowing Teacher to differ."""
    for result in (left, right):
        final = result["standalone_cartridge"]
        memory, control = final["memory"], final["original_question_controls"]
        train = result["training"]
        if (not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
            result["teacher_answer_in_final_prompt"] or final["teacher_answer_in_final_prompt"] or
            result["self_study"]["reference_answers_used"] or result["self_study"]["student_answer_hint_used"] or
            result["self_judge"]["applies_to"] != "full_context_diagnostic" or
            train["objective"] != "on-policy KL(teacher || student)" or
            train["selected_transfer_weight"] != 0 or train["value_basis_source"] != "document" or
            not train["teacher_conditioned_on_student_prefix"] or
            not train["student_rollouts_refreshed_each_step"] or not train["steps"] or
            result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0 or
            not memory["kv_only_materialized"] or memory["additional_source_kv_bytes_before_decode"] != 0 or
            memory["additional_training_adapter_bytes_before_decode"] != 0 or
            not control["same_selected_kv"] or not control["same_question_suffix"] or
            not control["same_generation_limit"] or control["teacher_answer_in_student_prompt"] or
            control["reference_answers_used"] or result["prediction"] != final["prediction"] or
            result["generated_token_ids"] != final["generated_token_ids"]):
            raise ValueError("A selected-evidence arm violates the original cartridge contract")
    if left["request_chat_date"] != right["request_chat_date"]:
        raise ValueError("The request chat dates differ")
    if any(left["selection"][key] != right["selection"][key] for key in SELECTION_FIELDS):
        raise ValueError("The selected source chunks or budget differ")
    if left["self_judge"]["native_prediction"] != right["self_judge"]["native_prediction"]:
        raise ValueError("The native full-context prediction differs")
    # The Teacher and zero-update tokens may change because the input token
    # sequence changes; treating equality as a prerequisite would filter out
    # exactly the effect this comparison is supposed to evaluate.
    return dict(same_teacher_tokens=left["teacher"]["tokens"] == right["teacher"]["tokens"],
                same_zero_tokens=left["standalone_cartridge"]["original_question_controls"]["zero"]["tokens"] ==
                                 right["standalone_cartridge"]["original_question_controls"]["zero"]["tokens"],
                same_practice_checks=left["self_study"]["checks"] == right["self_study"]["checks"])


def run_pair(engine, selector, request, config, *, time_budget_seconds=None):
    """Run both arms on the same frozen model; no reference answers are read."""
    if validate_configs(config, config) or not config.guard_enabled:
        raise ValueError("Identical original SD settings and a native full-context control are required")
    if {parameter.device for parameter in engine.model.parameters()} != {engine.device}:
        raise ValueError("The paired comparison requires a single-device model")
    stamp = engine.weight_stamp()
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state(engine.device) if engine.device.type == "cuda" else None
    deadline = None if time_budget_seconds is None else time.monotonic() + time_budget_seconds
    scope = getattr(engine.tokenizer, "request_scope", None)
    results = []
    try:
        with (scope() if callable(scope) else nullcontext()), original_question_controls():
            for variant in (False, True):
                torch.set_rng_state(cpu_rng)
                if gpu_rng is not None:
                    torch.cuda.set_rng_state(gpu_rng, engine.device)
                local = copy(engine)
                local.config = config
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("The paired request budget expired")
                context = (canonical_selected_evidence(engine.tokenizer, sink_tokens=config.sink_tokens)
                           if variant else nullcontext())
                with context:
                    results.append(MaskedTTT(local, selector).answer(request,
                        time_budget_seconds=remaining, compact_standalone=True))
        engine.assert_frozen(stamp)
        checks = validate_pair(*results)
    finally:
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, engine.device)
    return dict(left=results[0], right=results[1], changed_settings=["selected_evidence_tokenization"],
                same_loaded_model=True, same_device=True, same_gpu=engine.device.type == "cuda",
                same_request=True, independently_prefilled_caches=True, reference_answers_used=False,
                teacher_probability_tensors_compared=False, **checks)
