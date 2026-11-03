"""Selected-text ICL teacher -> on-policy joint KV SD -> fresh answer."""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace
import time

import torch

from .batched_sd import batched_teacher_logits
from .answer_selection import select_answer_arm
from .architectures import global_layers
from .memory import KVReadMemory
from .matched_cache_study import train_selected_cache
from .memory_accounting import (cache_tensors,cuda_allocated,cuda_peak,module_tensors,
                                optimizer_storage_bytes,storage_bytes)
from .model import Budget
from .on_policy import student_rollout
from .optimizer import distribution_kl
from .output_guard_runtime import guard_answers
from .reader import MemoryReader
from .runtime import private_forward, private_module_tree
from .selection import OpenAIChunkSelector
from .self_study_runtime import generate_from_cache,generate_self_study
from .standalone import answer_standalone
from .teacher import build_teacher, prefill_document, score_student_prefix
from .teacher_judge_runtime import judge_teacher
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
        teacher_judgment=judge_teacher(engine,teacher_reader,cache,teacher,working,layout,selected)
        teacher_judgment['enforced']=(engine.config.teacher_judge_policy!='off' and
                                     engine.config.teacher_judge_action=='enforce')
        use_teacher_signal=(teacher_judgment['apply_update'] or not teacher_judgment['enforced'])
        teacher_comparison=None
        if use_teacher_signal and engine.config.teacher_veto_against_native:
            # Compare two candidate answers under the same frozen full context.
            # A source-independent, unclear comparison cannot veto a signal.
            comparison_engine=copy(engine)
            comparison_engine.config=replace(engine.config,guard_policy='veto',
                guard_minimum_confidence=engine.config.teacher_veto_minimum_confidence,
                guard_minimum_label_mass=engine.config.teacher_veto_minimum_label_mass)
            teacher_comparison=guard_answers(comparison_engine,teacher_reader,cache,layout,working,
                                             native_prediction,teacher.prediction)
            use_teacher_signal=teacher_comparison['use_adapted']
        teacher_judgment['native_comparison']=teacher_comparison
        teacher_judgment['native_comparison_config']=(dict(policy='veto',
            minimum_confidence=engine.config.teacher_veto_minimum_confidence,
            minimum_label_mass=engine.config.teacher_veto_minimum_label_mass)
            if engine.config.teacher_veto_against_native else None)
        teacher_judgment['main_signal_accepted']=bool(use_teacher_signal)
        if not use_teacher_signal and not engine.config.practice_when_teacher_rejected:
            practice_paths=[]
            self_study=dict(generated_tokens=0,parsed=0,verified=0,checks=[],
                rejections={},reason='teacher_signal_not_selected',
                reference_answers_used=False,student_answer_hint_used=False)
        else:
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

        main_path=dict(name='main',student_suffix=suffix,teacher=teacher,
                       question=working.question)
        paths=([main_path] if use_teacher_signal else [])+practice_paths
        usable=engine.config.steps>0 and bool(paths)
        decision_state=('selected_signal' if usable and use_teacher_signal else
                        'verified_practice_only' if usable else
                        'teacher_signal_rejected' if not use_teacher_signal else 'zero_step_budget')
        support=dict(decision=dict(state=decision_state,
            planned_steps=engine.config.steps if usable else 0,apply_update=usable,
            teacher_main_signal_used=bool(usable and use_teacher_signal),
            verified_practice_paths=len(practice_paths),
            reference_answers_used=False),LLM_self_judge_enabled=engine.config.guard_enabled,
            reason='Selection applies to teacher source and trainable KV positions')
        if engine.config.matched_cache_direct:
            # The selected document cache is the same object for adaptation and
            # answer generation. The teacher signal and self-study paths are
            # exactly those used by the paired study, without the redundant
            # preliminary optimization on full-document KV.
            released=False;released_allocated_bytes=None
            if engine.config.release_full_cache_before_direct:
                # Full-document use has ended after the native answer, Teacher
                # judgment, and practice construction. The selected Teacher KV
                # was independently prefetched and remains available.
                if any(x._version != version or x.data_ptr() != pointer for x,version,pointer in cache_stamp):
                    raise AssertionError('The original document cache was changed')
                engine.assert_frozen(weights_before)
                allocated_before=cuda_allocated(engine.device)
                del cache_stamp,cache
                allocated_after=cuda_allocated(engine.device)
                if allocated_before is not None and allocated_after is not None:
                    released_allocated_bytes=max(0,allocated_before-allocated_after)
                released=True
            matched=train_selected_cache(engine,teacher_reader,teacher,layout,selected,
                                         suffix,paths,working,use_teacher_signal)
            if not released and any(x._version != version or x.data_ptr() != pointer for x,version,pointer in cache_stamp):
                raise AssertionError('The original document cache was changed')
            engine.assert_frozen(weights_before)
            chosen=select_answer_arm(engine.config.matched_cache_final_suffix,len(practice_paths),
                                     len(matched['steps']),bool(usable and use_teacher_signal))
            selected_answer=matched[chosen]
            source_bytes=storage_bytes(cache_tensors(teacher.document_cache))
            selected_bytes=matched['storage']['selected_kv_bytes']
            adapter_bytes=matched['storage']['adapter_bytes']
            standalone=dict(prediction=selected_answer['prediction'],selected_output_arm=chosen,
                generated_token_ids=selected_answer['generated_token_ids'],teacher_answer_in_final_prompt=False,
                memory=dict(source_selected_kv_bytes=source_bytes,materialized_selected_kv_bytes=selected_bytes,
                    cartridge_adapter_bytes=adapter_bytes,standalone_cartridge_bytes=selected_bytes+adapter_bytes,
                    document_tokens=matched['selected_document_tokens'],selected_chunks=matched['selected_chunks'],
                    raw_full_document_kv_used_at_decode=False,
                    selected_kv_representation='independently_prefilled_selected_text',
                    includes_generation_suffix_kv=False))
            updates=[dict(step,accepted=True) for step in matched['steps']]
            return dict(prediction=standalone['prediction'],generated_token_ids=standalone['generated_token_ids'],
                finish_reason=('eos' if standalone['generated_token_ids'][-1] in engine.eos_ids else 'length'),
                teacher=teacher.metadata(),teacher_judgment=teacher_judgment,self_study=self_study,
                self_judge=dict(native_prediction=native_prediction,adapted_prediction=standalone['prediction'],
                                final_choice='standalone_cartridge'),
                standalone_cartridge=standalone,matched_cache_study=matched,
                memory=dict(unit='bytes',model_resident_gpu_bytes=model_resident_bytes,
                    full_document_kv_bytes=original_kv_bytes,
                    full_cache_released_before_update=released,
                    allocated_bytes_released_before_update=released_allocated_bytes,
                    paired_run_peak_gpu_allocated_bytes=cuda_peak(engine.device),
                    gpu_peak_includes_model_and_training=True,
                    standalone_peak_is_not_measured_separately=True,
                    cache_sizes_exclude_question_and_generated_suffix=True),
                selection=prepared.metadata,support=support,
                training=dict(objective='on-policy KL(teacher || student)',
                    initial_kl=updates[0]['teacher_KL_before'] if updates else None,final_kl=None,
                    steps=updates,accepted_steps=len(updates),optimizer='Adam',
                    teacher_cap=engine.config.teacher_max_new_tokens,
                    student_rollout_limit=engine.config.student_rollout_tokens,
                    teacher_target_scope=engine.config.teacher_target_scope,
                    student_rollouts_refreshed_each_step=True,teacher_conditioned_on_student_prefix=True,
                    verified_practice_questions=len(practice_paths),
                    teacher_main_signal_used=bool(usable and use_teacher_signal),
                    adapted_layers=global_layers(engine.model.config),checkpointed_layers=True,
                    causal_batched_teacher_forcing=True,training_cache='selected_document'),
                timing=dict(ttft_seconds=None,prefill_seconds=prefill_seconds,
                    teacher_seconds=teacher.elapsed_seconds,adaptation_and_decoding_seconds=matched['elapsed_seconds'],
                    total_seconds=time.perf_counter()-started,
                    scope='Complete request until all paired outputs; individual TTFT is not measured',
                    selection_backend=prepared.metadata.get('backend','online_decisions'),
                    offline_selection_seconds=prepared.metadata.get('offline_scoring_seconds',0),
                    offline_selection_cost_usd=prepared.metadata.get('offline_scoring_cost_usd',0),
                    offline_selection_in_ttft=False),
                backbone_unchanged=True,original_document_KV_unchanged=True,
                final_question_recomputed=True,teacher_answer_in_final_prompt=False)
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
        began=time.perf_counter()
        if usable:
            memory.prepare_value_basis(cache,memory.adapted_layers)
            memory.value_enabled=True
        engine.synchronize();setup_seconds=time.perf_counter()-began
        initial_kl=final_kl=None;steps=[];positions=[];optimizer_bytes=0
        use_choice=(use_teacher_signal and engine.config.teacher_target_scope=='choice_or_all'
                    and 'Choices:' in working.question)
        use_adaptive_open=(use_teacher_signal and not use_choice and engine.config.adaptive_open_ended)
        teacher_version=teacher.probabilities._version
        began=time.perf_counter()
        if usable:
            memory.native_mode=False
            # A veto applies only to the original-question teacher signal.
            # Independently verified practice paths remain eligible and share
            # the same request-local memory without seeing the rejected answer.
            required_steps=(engine.config.steps+len(practice_paths) if use_teacher_signal
                            else max(engine.config.steps,len(practice_paths)))
            params=(memory.gain,memory.value_coefficients)
            optimizer=torch.optim.Adam([
                dict(params=[params[0]],lr=engine.config.gain_learning_rate),
                dict(params=[params[1]],lr=engine.config.value_learning_rate)],foreach=False)
            try:
                max_steps=(max(required_steps,engine.config.max_answer_steps)
                           if use_choice or use_adaptive_open else required_steps)
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
                        stop_choice=(use_choice and (not positions or before<=engine.config.answer_KL_stop))
                        stop_open=(use_adaptive_open and before<=engine.config.adaptive_open_ended_kl_stop)
                        if index>=required_steps and (stop_choice or stop_open):
                            final_kl=before
                            del logits,loss
                            break
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
                    steps.append(dict(step=index+1,accepted=True,teacher_KL_before=before,
                        question_path=path['name'],
                        student_rollout_tokens=rollout.tolist(),
                        target_positions=positions or list(range(len(rollout))),
                        gain_gradient_norm=norms[0],value_gradient_norm=norms[1],gradient_norm=total))
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
        standalone=(answer_standalone(engine,layout,selected,teacher,memory,working.max_new_tokens)
                    if engine.config.standalone_cartridge_enabled else None)
        matched_cache=(train_selected_cache(engine,teacher_reader,teacher,layout,selected,
                       suffix,paths,working,use_teacher_signal)
                       if engine.config.matched_cache_probe else None)
        engine.synchronize()
        total_peak_bytes=cuda_peak(engine.device)

        if any(
            x._version != version or x.data_ptr() != pointer for x, version, pointer in cache_stamp
        ):
            raise AssertionError("The original document cache was changed")
        engine.assert_frozen(weights_before)
        return dict(
            prediction=engine.tokenizer.decode(generated, skip_special_tokens=True),
            generated_token_ids=generated,
            finish_reason="eos" if generated[-1] in engine.eos_ids else "length",
            teacher=teacher.metadata(),
            teacher_judgment=teacher_judgment,
            self_study=self_study,
            self_judge=dict(guard,native_prediction=native_prediction,
                            adapted_prediction=adapted_prediction),
            standalone_cartridge=standalone,
            matched_cache_study=matched_cache,
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
            ),
            selection=prepared.metadata,
            support=support,
            training=dict(
                objective="on-policy KL(teacher || student)",
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
                kl_initial_and_final_may_use_different_student_prefixes=True,
                kl_initial_and_final_may_use_different_questions=bool(practice_paths),
                verified_practice_questions=len(practice_paths),
                teacher_main_signal_used=bool(usable and use_teacher_signal),
                practice_when_teacher_rejected=engine.config.practice_when_teacher_rejected,
                teacher_veto_against_native=engine.config.teacher_veto_against_native,
                adaptive_max_answer_steps=engine.config.max_answer_steps,
                adaptive_answer_KL_stop=engine.config.answer_KL_stop,
                adaptive_open_ended=use_adaptive_open,
                adaptive_open_ended_kl_stop=engine.config.adaptive_open_ended_kl_stop,
                adapted_layers=memory.adapted_layers,
                checkpointed_layers=True,causal_batched_teacher_forcing=True,
            ),
            timing=dict(
                ttft_seconds=ttft,
                prefill_seconds=prefill_seconds,
                teacher_seconds=teacher.elapsed_seconds,
                teacher_judgment_seconds=teacher_judgment['seconds'],
                teacher_native_comparison_seconds=(teacher_comparison['seconds']
                    if teacher_comparison is not None else 0.0),
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
                scope=(
                    "Online request including selection, native answer, Teacher, self-study, adaptation, "
                    "candidate generation and full-context guard through the first final token; excludes model loading, "
                    "queue time, and previously completed shared offline selection"
                ),
            ),
            backbone_unchanged=True,
            original_document_KV_unchanged=True,
            final_question_recomputed=True,
            teacher_answer_in_final_prompt=False,
        )
