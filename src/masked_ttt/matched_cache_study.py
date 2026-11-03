"""Paired zero-update / on-policy SD on the exact standalone document cache.

This is an experimental comparison, not a change to the archived pipeline.
The selected teacher document KV is immutable. Both outputs use the same
student suffix and generation budget; no reference answer enters inference.
"""
import time

import torch

from ._layout import selected_read_plan
from .batched_sd import batched_teacher_logits
from .memory import KVReadMemory
from .memory_accounting import cache_tensors, module_tensors, storage_bytes
from .on_policy import student_rollout
from .optimizer import distribution_kl
from .reader import MemoryReader
from .standalone import selected_chunk_positions
from .teacher import score_student_prefix
from .teacher_targeting import choice_positions


def _plan(engine, layout, selected, teacher, original_suffix):
    separator = engine.tokenizer.encode('\n\n', add_special_tokens=False)
    plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, selected, separator)
    if (plan.document_end != teacher.document_end or
            tuple(plan.ids[:plan.document_end]) != teacher.document_ids or
            tuple(plan.ids[plan.document_end:]) != tuple(original_suffix.tolist())):
        raise ValueError('Student selected document and question must match final inference')
    chunks = selected_chunk_positions(layout, selected, plan, len(separator))
    return plan, chunks


def _prediction(engine, reader, cache, suffix, limit):
    tokens = student_rollout(engine, reader, cache, suffix, limit)
    return dict(prediction=engine.tokenizer.decode(tokens.tolist(), skip_special_tokens=True),
                generated_token_ids=tokens.tolist())


def train_selected_cache(engine, teacher_reader, teacher, layout, selected, suffix,
                         paths, working, use_teacher_signal):
    """Independently optimize the cache that the final standalone answer uses."""
    began = time.perf_counter()
    plan, chunks = _plan(engine, layout, selected, teacher, suffix)
    selected_cache = teacher.document_cache
    if selected_cache is None:
        raise ValueError('A frozen selected document cache is required')
    source_tensors = cache_tensors(selected_cache)
    stamp = [(x, x._version, x.data_ptr()) for x in source_tensors]
    memory = KVReadMemory(engine.model.config, chunks, list(range(len(chunks))),
        plan.document_end, rank=engine.config.value_rank, value_radius=engine.config.value_radius,
        gain_bound=engine.config.gain_bound, sink_tokens=engine.config.sink_tokens,
        seed=engine.config.seed).to(engine.device)
    reader = MemoryReader(engine, memory)
    memory.select_actual_read_path()
    zero = _prediction(engine, reader, selected_cache, suffix, working.max_new_tokens)
    zero['reference_answers_used'] = False
    zero['teacher_answer_in_final_prompt'] = False
    # The teacher suffix contains the same question and output-format request,
    # not the teacher's generated answer. Decode both suffixes against the
    # exact same selected cache before and after the same update.
    format_suffix = teacher.question_tokens
    if format_suffix is None or format_suffix.ndim != 1 or not len(format_suffix):
        raise ValueError('The teacher question suffix is unavailable')
    zero_format = _prediction(engine, reader, selected_cache, format_suffix, working.max_new_tokens)
    zero_format['reference_answers_used'] = False
    zero_format['teacher_answer_in_final_prompt'] = False
    usable = engine.config.steps > 0 and bool(paths)
    updates = []
    if usable:
        memory.prepare_value_basis(selected_cache, memory.adapted_layers)
        memory.value_enabled = True
        memory.native_mode = False
        practice_count = sum(path['name'] != 'main' for path in paths)
        required_steps = (engine.config.steps + practice_count if use_teacher_signal
                          else max(engine.config.steps, practice_count))
        use_choice = (use_teacher_signal and engine.config.teacher_target_scope == 'choice_or_all'
                      and 'Choices:' in working.question)
        use_adaptive_open = (use_teacher_signal and not use_choice and engine.config.adaptive_open_ended)
        maximum_steps = (max(required_steps, engine.config.max_answer_steps)
                         if use_choice or use_adaptive_open else required_steps)
        parameters = (memory.gain, memory.value_coefficients)
        optimizer = torch.optim.Adam([
            dict(params=[parameters[0]], lr=engine.config.gain_learning_rate),
            dict(params=[parameters[1]], lr=engine.config.value_learning_rate)], foreach=False)
        try:
            for index in range(maximum_steps):
                engine.budget.check()
                optimizer.zero_grad(set_to_none=True)
                path = paths[index % len(paths)] if index < required_steps else paths[0]
                current_suffix, current_teacher = path['student_suffix'], path['teacher']
                rollout = student_rollout(engine, reader, selected_cache, current_suffix,
                    min(working.max_new_tokens, engine.config.student_rollout_tokens))
                targets = score_student_prefix(engine, teacher_reader, current_teacher, rollout)
                main_choice = use_choice and path['name'] == 'main'
                positions = choice_positions(engine.tokenizer, rollout.tolist()) if main_choice else []
                indices = (torch.tensor(positions, device=rollout.device, dtype=torch.long)
                           if positions else None)
                target = targets.index_select(0, indices) if indices is not None else targets
                with torch.enable_grad():
                    logits = batched_teacher_logits(engine, reader, selected_cache, current_suffix,
                                                   rollout, target_positions=indices)
                    loss = distribution_kl(logits, target)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite matched-cache teacher KL')
                    before = float(loss.detach())
                    stop_choice = use_choice and (not positions or before <= engine.config.answer_KL_stop)
                    stop_open = use_adaptive_open and before <= engine.config.adaptive_open_ended_kl_stop
                    if index >= required_steps and (stop_choice or stop_open):
                        break
                    loss.backward()
                if any(p.grad is None for p in parameters):
                    raise AssertionError('Both selected KV parameter groups need gradients')
                norms = [float(p.grad.float().norm()) for p in parameters]
                if not all(torch.isfinite(torch.tensor(norms))):
                    raise FloatingPointError('Nonfinite matched-cache gradient')
                torch.nn.utils.clip_grad_norm_(parameters, engine.config.gradient_clip,
                                               error_if_nonfinite=True)
                optimizer.step()
                memory.constrain()
                updates.append(dict(step=index + 1, question_path=path['name'],
                    teacher_KL_before=before, gain_gradient_norm=norms[0],
                    value_gradient_norm=norms[1], rollout_length=len(rollout),
                    student_rollout_tokens=rollout.tolist(),
                    target_positions=positions or list(range(len(rollout)))))
        finally:
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    memory.select_actual_read_path()
    trained = _prediction(engine, reader, selected_cache, suffix, working.max_new_tokens)
    trained['reference_answers_used'] = False
    trained['teacher_answer_in_final_prompt'] = False
    trained_format = _prediction(engine, reader, selected_cache, format_suffix, working.max_new_tokens)
    trained_format['reference_answers_used'] = False
    trained_format['teacher_answer_in_final_prompt'] = False
    with torch.no_grad():
        branch = reader.compiled_branch(selected_cache)
        storage = dict(selected_kv_bytes=storage_bytes(cache_tensors(branch)),
                       adapter_bytes=storage_bytes(module_tensors(memory)))
    if any(x._version != version or x.data_ptr() != pointer for x, version, pointer in stamp):
        raise AssertionError('The selected document cache was modified')
    engine.synchronize()
    return dict(zero_update=zero, trained=trained, zero_format=zero_format,
        trained_format=trained_format, steps=updates,
        accepted_steps=len(updates), selected_chunks=len(chunks),
        selected_document_tokens=plan.document_end, storage=storage,
        same_document_cache_for_training_and_generation=True,
        same_suffix_and_budget_for_zero_and_trained=True,
        same_document_cache_and_budget_for_format_comparison=True,
        teacher_target_on_student_prefix=True, reference_answers_used=False,
        source_selected_cache_unchanged=True, elapsed_seconds=time.perf_counter() - began)
