"""Read-only check of native versus zero-correction materialized KV decoding."""

import torch


@torch.no_grad()
def measure_zero_path_parity(engine, layout, selected, teacher, trained, *, max_new_tokens=8):
    """Compare the same selected cache with both execution paths at zero state.

    The augmented key has its extra score dimension set to zero and the Value
    adjustment is zero, so the mathematical attention function is unchanged.
    Actual kernels may use different padded dimensions and numerical precision.
    Only a new local reader is zeroed; the trained state and source KV are read
    only. The comparison does not use the answer label or teacher answer.
    """
    from masked_ttt.memory_accounting import cache_tensors
    from masked_ttt.runtime import private_forward
    from masked_ttt.standalone import build_standalone

    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("A positive decoding budget is required")
    source = teacher.document_cache
    if source is None:
        raise ValueError("The independently prefetched selected cache is required")
    stamps = [(tensor, tensor._version, tensor.data_ptr()) for tensor in cache_tensors(source)]
    reader, unused_branch, suffix, length, _ = build_standalone(engine, layout, selected, teacher, trained)
    del unused_branch
    if length + len(suffix) + max_new_tokens > engine.config.max_context:
        raise ValueError("The parity probe exceeds the model context")
    reader.memory.gain.zero_()
    reader.memory.value_coefficients.zero_()
    reader.memory.value_enabled = True

    def decode(force_augmented):
        reader.memory.native_mode = not force_augmented
        branch = reader.compiled_branch(source)
        output = private_forward(reader.reader, suffix, length, branch)
        logits = engine.model.lm_head(output.last_hidden_state[0, -1:])[0].float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite parity-probe logits")
        first = logits
        tokens = []
        while len(tokens) < max_new_tokens:
            token = int(logits.argmax())
            tokens.append(token)
            if token in engine.eos_ids or len(tokens) == max_new_tokens:
                break
            engine.budget.check()
            output = private_forward(reader.reader, torch.tensor([token], device=engine.device),
                                     length + len(suffix) + len(tokens) - 1, branch)
            logits = engine.model.lm_head(output.last_hidden_state[0, -1:])[0].float()
        return first, tokens

    native, native_tokens = decode(False)
    augmented, augmented_tokens = decode(True)
    if not bool(torch.isfinite(native).all() and torch.isfinite(augmented).all()):
        raise FloatingPointError("Nonfinite parity-probe logits")
    native_logp, augmented_logp = native.log_softmax(-1), augmented.log_softmax(-1)
    difference = native - augmented
    top = native.topk(min(2, native.numel())).values
    if any(tensor._version != version or tensor.data_ptr() != pointer for tensor, version, pointer in stamps):
        raise AssertionError("The selected source KV changed during zero-path comparison")
    return dict(
        probe="same independently prefetched selected cache, original question, zero correction",
        first_token_kl=float((native_logp.exp() * (native_logp - augmented_logp)).sum().clamp_min(0)),
        first_logit_max_absolute_difference=float(difference.abs().max()),
        first_logit_rms_difference=float(difference.square().mean().sqrt()),
        first_token_top1_changed=bool(native.argmax() != augmented.argmax()),
        native_first_token_margin=float(top[0] - top[1]) if len(top) > 1 else None,
        generated_tokens_changed=native_tokens != augmented_tokens,
        native_generated_count=len(native_tokens), augmented_generated_count=len(augmented_tokens),
        teacher_answer_used=False, reference_answers_used=False, source_cache_modified=False,
    )
