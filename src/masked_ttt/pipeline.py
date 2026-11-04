"""Selected-text ICL teacher -> on-policy joint KV SD -> fresh answer."""

from copy import copy
from types import SimpleNamespace
import gc
import time
import weakref

import torch

from .batched_sd import batched_teacher_logits
from .memory import KVReadMemory
from .memory_accounting import (cache_tensors,cuda_allocated,cuda_peak,module_tensors,
                                optimizer_storage_bytes,storage_bytes)
from .model import Budget
from .on_policy import student_rollout
from .optimizer import distribution_kl
from .output_guard_runtime import guard_answers
from .reader import MemoryReader
from .runtime import private_forward, private_module_tree
from .selection import OpenAIChunkSelector
from .selected_transfer import selected_transfer_loss
from .self_study_runtime import generate_from_cache,generate_self_study
from .shared_selected import shared_selected_reader
from .standalone import answer_standalone
from .teacher import build_teacher, prefill_document, score_student_prefix
from .teacher_targeting import choice_positions


class MaskedTTT:
    def __init__(self, backbone, selector=None):
        self.backbone = backbone
        self.selector = selector or OpenAIChunkSelector()

    def answer(self, request, *, time_budget_seconds=None):
        started = time.perf_counter()
        # The backbone tensors are shared; the deadline, readers, KV, and optimizer
        # belong to this request. No adaptation state is stored on self.
        engine = copy(self.backbone)
        engine.budget = Budget(time_budget_seconds)
        engine.synchronize()
        if engine.device.type=='cuda':torch.cuda.reset_peak_memory_stats(engine.device)
        model_resident_bytes=cuda_allocated(engine.device)
        weights_before = engine.weight_stamp()
        prepared = self.selector.prepare(engine, request)
        working, layout, selected = prepared.request, prepared.layout, prepared.selected
        if not selected or selected != sorted(set(selected)):
            raise ValueError("A chronological nonempty selected chunk set is required")
        if len(layout.ids) + working.max_new_tokens > engine.config.max_context:
            raise ValueError("The prepared request exceeds the model context")
        ids = torch.tensor(layout.ids, device=engine.device)
        length, suffix = layout.document_end, ids[layout.document_end :]
        teacher_reader = SimpleNamespace(
            model=private_module_tree(engine.model.model), lm_head=engine.model.lm_head
        )
        teacher_request_setup_seconds = time.perf_counter() - started

        began = time.perf_counter()
        cache = prefill_document(
            engine, engine.model, layout.ids, layout.initial, [c[-1] + 1 for c in layout.chunks]
        )
        prefill_seconds = time.perf_counter() - began
        original_kv_bytes=storage_bytes(cache_tensors(cache))
        cache_stamp = [
            (x, x._version, x.data_ptr()) for item in cache.layers for x in (item.keys, item.values)
        ]
        native_tokens=native_prediction=None
        if engine.config.guard_enabled:
            native_tokens,native_prediction=generate_from_cache(
                engine,teacher_reader,cache,length,suffix,working.max_new_tokens)
        teacher = build_teacher(engine, teacher_reader, working, layout, selected)
        try:
            practice_paths,self_study=generate_self_study(
                engine,teacher_reader,teacher,working,layout,selected)
        except ValueError as exc:
            # A non-reusable token boundary is an explicit no-practice case;
            # no altered document cache is silently substituted.
            practice_paths=[]
            self_study=dict(generated_tokens=0,parsed=0,verified=0,checks=[],
                rejections={'unusable_practice_prefix':1},reason=str(exc),
                reference_answers_used=False,student_answer_hint_used=False)

        memory = KVReadMemory(
            engine.model.config,
            layout.chunks,
            selected,
            length,
            rank=engine.config.value_rank,
            value_radius=engine.config.value_radius,
            gain_bound=engine.config.gain_bound,
            sink_tokens=engine.config.sink_tokens,
            seed=engine.config.seed,
        ).to(engine.device)
        reader = MemoryReader(engine, memory)
        usable=engine.config.steps>0
        support=dict(decision=dict(state='selected_signal' if usable else 'empty_teacher',
            planned_steps=engine.config.steps if usable else 0,apply_update=usable,
            reference_answers_used=False),LLM_self_judge_enabled=engine.config.guard_enabled,
            reason='Selection applies to teacher source and trainable KV positions')
        began=time.perf_counter()
        if usable:
            memory.value_enabled=True
            if engine.config.value_basis_source=="document":
                memory.prepare_value_basis(cache,memory.adapted_layers)
            else:
                # The frozen low-rank input basis is aligned with the cache on
                # which the learned state will be materialized. Only B and g
                # are trained, still on the full-document student path.
                memory.active_value_layers=set(memory.adapted_layers)
                memory.value_layers.zero_()
                memory.value_layers[memory.adapted_layers]=True
                basis_view=shared_selected_reader(engine,layout,selected,teacher,memory)
                basis_view.memory.prepare_value_basis(teacher.document_cache,memory.adapted_layers)
                del basis_view
        selected_reader=None;selected_cache_stamp=[]
        if usable and engine.config.selected_transfer_weight>0:
            selected_reader=shared_selected_reader(engine,layout,selected,teacher,memory)
            selected_cache_stamp=[(x,x._version,x.data_ptr()) for x in cache_tensors(teacher.document_cache)]
        engine.synchronize();setup_seconds=time.perf_counter()-began
        initial_kl=final_kl=None;steps=[];positions=[];optimizer_bytes=0
        use_choice=(engine.config.teacher_target_scope=='choice_or_all' and 'Choices:' in working.question)
        teacher_version=teacher.probabilities._version
        began=time.perf_counter()
        if usable:
            memory.native_mode=False
            paths=[dict(name='main',student_suffix=suffix,teacher=teacher,
                        question=working.question)]+practice_paths
            required_steps=engine.config.steps+len(practice_paths)
            params=(memory.gain,memory.value_coefficients)
            optimizer=torch.optim.Adam([
                dict(params=[params[0]],lr=engine.config.gain_learning_rate),
                dict(params=[params[1]],lr=engine.config.value_learning_rate)],foreach=False)
            try:
                max_steps=max(required_steps,engine.config.max_answer_steps) if use_choice else required_steps
                for index in range(max_steps):
                    engine.budget.check();optimizer.zero_grad(set_to_none=True)
                    path=paths[index%len(paths)] if index<required_steps else paths[0]
                    current_suffix,current_teacher=path['student_suffix'],path['teacher']
                    # Refresh from the current student on every update. The teacher
                    # then scores exactly these prefixes, never its own answer.
                    rollout=student_rollout(engine,reader,cache,current_suffix,
                        min(working.max_new_tokens,engine.config.student_rollout_tokens))
                    teacher_probs=score_student_prefix(engine,teacher_reader,current_teacher,rollout)
                    main_choice=use_choice and path['name']=='main'
                    positions=choice_positions(engine.tokenizer,rollout.tolist()) if main_choice else []
                    target_positions=(torch.tensor(positions,device=rollout.device,dtype=torch.long)
                                      if positions else None)
                    target=(teacher_probs.index_select(0,target_positions) if positions else teacher_probs)
                    with torch.enable_grad():
                        logits=batched_teacher_logits(engine,reader,cache,current_suffix,rollout,
                                                      target_positions=target_positions)
                        loss=distribution_kl(logits,target)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite teacher KL')
                        before=float(loss.detach())
                        if index>=required_steps and (not positions or before<=engine.config.answer_KL_stop):
                            final_kl=before
                            del logits,loss
                            break
                        transfer_kl=None;transfer_tokens=None;transfer_positions=None
                        if selected_reader is not None:
                            transfer_loss,transfer_rollout,transfer_positions=selected_transfer_loss(
                                engine,reader,selected_reader,cache,teacher.document_cache,current_suffix,
                                min(working.max_new_tokens,engine.config.student_rollout_tokens),
                                target_choices=main_choice)
                            transfer_kl=float(transfer_loss.detach())
                            transfer_tokens=transfer_rollout.tolist()
                            loss=loss+engine.config.selected_transfer_weight*transfer_loss
                        loss.backward()
                    if any(parameter.grad is None for parameter in params):
                        raise AssertionError('Both selected KV parameter groups need gradients')
                    if bool(params[0].grad[:,:,~memory.selected_chunk_mask].any()):
                        raise AssertionError('An unselected chunk received a gain gradient')
                    norms=[float(parameter.grad.float().norm()) for parameter in params]
                    total=float(torch.nn.utils.clip_grad_norm_(params,engine.config.gradient_clip,
                                                             error_if_nonfinite=True))
                    if initial_kl is None:initial_kl=before
                    optimizer.step();memory.constrain()
                    optimizer_bytes=max(optimizer_bytes,optimizer_storage_bytes(optimizer))
                    step=dict(step=index+1,accepted=True,teacher_KL_before=before,
                        question_path=path['name'],
                        student_rollout_tokens=rollout.tolist(),
                        target_positions=positions or list(range(len(rollout))),
                        gain_gradient_norm=norms[0],value_gradient_norm=norms[1],gradient_norm=total)
                    if transfer_kl is not None:
                        step.update(selected_transfer_KL_before=transfer_kl,
                                    selected_rollout_tokens=transfer_tokens,
                                    selected_target_positions=transfer_positions)
                    steps.append(step)
                if final_kl is None:
                    with torch.no_grad():
                        final_kl=float(distribution_kl(
                            batched_teacher_logits(engine,reader,cache,current_suffix,rollout,
                                                   target_positions=target_positions),target))
                if not bool(torch.isfinite(torch.tensor(final_kl))):
                    raise FloatingPointError('Nonfinite final teacher KL')
                if teacher.probabilities._version!=teacher_version:
                    raise AssertionError('Teacher target was modified')
            finally:
                optimizer.zero_grad(set_to_none=True);optimizer.state.clear()
        engine.synchronize();adaptation_seconds=time.perf_counter()-began
        support['decision']['planned_steps']=len(steps)

        for parameter in memory.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        memory.select_actual_read_path()
        # This branch contains only the original document plus final adapters.
        # Every question/answer position used for training is recomputed below.
        with torch.no_grad():
            branch = reader.compiled_branch(cache)
            full_materialized_kv_bytes=storage_bytes(cache_tensors(branch))
            full_decode_state_bytes=storage_bytes(cache_tensors(cache)+cache_tensors(branch)+
                                                   module_tensors(memory))
            output = private_forward(reader.reader, suffix, length, branch)
            generated = [int(engine.model.lm_head(output.last_hidden_state[0, -1:])[0].argmax())]
            engine.synchronize()
            candidate_first_token_seconds = time.perf_counter() - started
            while len(generated) < working.max_new_tokens and generated[-1] not in engine.eos_ids:
                engine.budget.check()
                output = private_forward(
                    reader.reader,
                    torch.tensor([generated[-1]], device=engine.device),
                    len(layout.ids) + len(generated) - 1,
                    branch,
                )
                generated.append(
                    int(engine.model.lm_head(output.last_hidden_state[0, -1:])[0].argmax())
                )
            engine.synchronize()

        adapted_prediction=engine.tokenizer.decode(generated,skip_special_tokens=True)
        if engine.config.guard_enabled:
            guard=guard_answers(engine,teacher_reader,cache,layout,working,native_prediction,adapted_prediction)
            if not guard['use_adapted']:
                generated=native_tokens
            # The final answer cannot be streamed until comparison completes.
            engine.synchronize();ttft=time.perf_counter()-started
        else:
            guard=dict(use_adapted=True,reason='guard_disabled',reviews=[],seconds=0.,
                       reference_answers_used=False)
            ttft=candidate_first_token_seconds

        engine.synchronize()
        guarded_peak_bytes=cuda_peak(engine.device)
        adapter_bytes=storage_bytes(module_tensors(memory))
        full_plus_adapter_bytes=storage_bytes(cache_tensors(cache)+module_tensors(memory))
        if any(x._version != version or x.data_ptr() != pointer for x,version,pointer in cache_stamp):
            raise AssertionError("The original document cache was changed")
        if any(x._version != version or x.data_ptr() != pointer for x,version,pointer in selected_cache_stamp):
            raise AssertionError("The original selected evidence cache was changed")
        release_info=dict(requested=engine.config.release_full_before_cartridge,
                          full_cache_tensors_live_before_standalone=None,
                          cuda_allocated_before_release_bytes=None,
                          cuda_allocated_after_release_bytes=None)
        if engine.config.release_full_before_cartridge:
            # Preserve only weak references for validation; stamps, the full
            # decode branch and the final training graph otherwise retain KV.
            full_refs=[weakref.ref(t) for t in cache_tensors(cache)]
            engine.synchronize()
            release_info["cuda_allocated_before_release_bytes"]=cuda_allocated(engine.device)
            cache_stamp=None
            branch=None
            output=None
            cache=None
            logits=loss=target=teacher_probs=None
            transfer_loss=transfer_rollout=None
            gc.collect()
            remaining=sum(ref() is not None for ref in full_refs)
            release_info["full_cache_tensors_live_before_standalone"]=remaining
            if remaining:
                raise AssertionError("Full document cache remains live before standalone decoding")
            if engine.device.type=='cuda':
                torch.cuda.empty_cache()
            engine.synchronize()
            release_info["cuda_allocated_after_release_bytes"]=cuda_allocated(engine.device)
        standalone_started=time.perf_counter()
        standalone=(answer_standalone(engine,layout,selected,teacher,memory,working.max_new_tokens)
                    if engine.config.standalone_cartridge_enabled else None)
        engine.synchronize()
        total_peak_bytes=cuda_peak(engine.device)

        if any(x._version != version or x.data_ptr() != pointer for x,version,pointer in selected_cache_stamp):
            raise AssertionError("The original selected evidence cache was changed")
        engine.assert_frozen(weights_before)
        full_prediction=engine.tokenizer.decode(generated,skip_special_tokens=True)
        full_finish_reason="eos" if generated[-1] in engine.eos_ids else "length"
        if standalone is not None:
            final_tokens=standalone['generated_token_ids']
            final_prediction=standalone['prediction']
            source='standalone_cartridge'
            standalone_first=standalone.get('first_token_seconds')
            final_ttft=(standalone_started-started+standalone_first
                        if standalone_first is not None else None)
            ttft_scope=('Online request through the first standalone Cartridge token; '
                        'includes selected-cache materialization, excludes model loading and offline scoring')
        else:
            final_tokens=generated
            final_prediction=full_prediction
            source='full_context'
            final_ttft=ttft
            ttft_scope=('Online request through the first final full-context token, including '
                        'any full-context guard; excludes model loading and offline scoring')
        if not final_tokens:raise AssertionError('The primary output contains no tokens')
        return dict(
            prediction=final_prediction,
            generated_token_ids=final_tokens,
            finish_reason="eos" if final_tokens[-1] in engine.eos_ids else "length",
            output_source=source,
            full_context=dict(prediction=full_prediction,generated_token_ids=generated,
                              finish_reason=full_finish_reason),
            teacher=teacher.metadata(),
            self_study=self_study,
            self_judge=dict(guard,applies_to=('full_context_diagnostic' if standalone is not None else 'primary_output'),
                            native_prediction=native_prediction,
                            adapted_prediction=adapted_prediction),
            standalone_cartridge=standalone,
            memory=dict(
                unit='bytes',model_resident_gpu_bytes=model_resident_bytes,
                full_document_kv_bytes=original_kv_bytes,
                full_adapter_bytes=adapter_bytes,
                full_kv_plus_adapter_bytes=full_plus_adapter_bytes,
                full_materialized_decode_kv_bytes=full_materialized_kv_bytes,
                full_decode_state_including_original_kv_bytes=full_decode_state_bytes,
                optimizer_state_peak_bytes=optimizer_bytes,
                full_request_peak_gpu_allocated_bytes=guarded_peak_bytes,
                paired_run_peak_gpu_allocated_bytes=total_peak_bytes,
                gpu_peak_includes_model_and_training=True,
                standalone_peak_is_not_measured_separately=True,
                cache_sizes_exclude_question_and_generated_suffix=True,
                full_cache_release_before_standalone=release_info,
            ),
            selection=prepared.metadata,
            support=support,
            training=dict(
                objective=("on-policy KL(selected teacher || full student) + weighted "
                           "KL(detached full student || selected student)"
                           if selected_reader is not None else "on-policy KL(teacher || student)"),
                initial_kl=initial_kl,
                final_kl=final_kl,
                steps=steps,
                accepted_steps=sum(r["accepted"] for r in steps),
                optimizer='Adam',teacher_cap=engine.config.teacher_max_new_tokens,
                teacher_prefix_truncated=not teacher.complete,
                teacher_target_scope=engine.config.teacher_target_scope,
                teacher_target_positions=positions or list(range(len(rollout))) if usable else [],
                student_rollout_limit=engine.config.student_rollout_tokens,
                student_rollouts_refreshed_each_step=True,
                teacher_conditioned_on_student_prefix=True,
                selected_transfer_weight=engine.config.selected_transfer_weight,
                value_basis_source=engine.config.value_basis_source,
                selected_transfer_uses_selected_student_prefix=selected_reader is not None,
                selected_transfer_full_target_detached=selected_reader is not None,
                kl_initial_and_final_may_use_different_student_prefixes=True,
                kl_initial_and_final_may_use_different_questions=bool(practice_paths),
                verified_practice_questions=len(practice_paths),
                adaptive_max_answer_steps=engine.config.max_answer_steps,
                adaptive_answer_KL_stop=engine.config.answer_KL_stop,
                adapted_layers=memory.adapted_layers,
                checkpointed_layers=True,causal_batched_teacher_forcing=True,
            ),
            timing=dict(
                ttft_seconds=final_ttft,
                full_context_ttft_seconds=ttft,
                prefill_seconds=prefill_seconds,
                teacher_seconds=teacher.elapsed_seconds,
                icl_teacher_first_token_seconds_segmented=(
                    teacher_request_setup_seconds + teacher.first_token_seconds),
                icl_teacher_timing_scope=(
                    'Sum of shared request setup and selected Teacher planning, prefill, and '
                    'first token; excludes intervening Student document prefill and offline selection'
                ),
                support_seconds=0.0,adapter_setup_seconds=setup_seconds,
                adaptation_seconds=adaptation_seconds,
                candidate_first_token_seconds=candidate_first_token_seconds,
                full_context_guard_seconds=guard['seconds'],
                total_seconds=time.perf_counter() - started,
                selection_backend=prepared.metadata.get("backend", "online_decisions"),
                offline_selection_seconds=prepared.metadata.get("offline_scoring_seconds", 0),
                offline_selection_cost_usd=prepared.metadata.get("offline_scoring_cost_usd", 0),
                offline_selection_in_ttft=False,
                scope=ttft_scope,
            ),
            backbone_unchanged=True,
            original_document_KV_unchanged=True,
            final_question_recomputed=True,
            teacher_answer_in_final_prompt=False,
        )
