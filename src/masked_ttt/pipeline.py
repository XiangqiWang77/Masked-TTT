"""Selected-text ICL teacher -> on-policy joint KV SD -> fresh answer."""

from copy import copy
from types import SimpleNamespace
import time

import torch

from .batched_sd import batched_teacher_logits
from .memory import KVReadMemory
from .model import Budget
from .on_policy import student_rollout
from .optimizer import distribution_kl
from .reader import MemoryReader
from .runtime import private_forward, private_module_tree
from .selection import OpenAIChunkSelector
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

        began = time.perf_counter()
        cache = prefill_document(
            engine, engine.model, layout.ids, layout.initial, [c[-1] + 1 for c in layout.chunks]
        )
        prefill_seconds = time.perf_counter() - began
        cache_stamp = [
            (x, x._version, x.data_ptr()) for item in cache.layers for x in (item.keys, item.values)
        ]
        teacher = build_teacher(engine, teacher_reader, working, layout, selected)

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
            reference_answers_used=False),LLM_self_judge_enabled=False,
            reason='Selection applies to teacher source and trainable KV positions')
        began=time.perf_counter()
        if usable:
            memory.prepare_value_basis(cache,memory.adapted_layers)
            memory.value_enabled=True
        engine.synchronize();setup_seconds=time.perf_counter()-began
        initial_kl=final_kl=None;steps=[];positions=[]
        use_choice=(engine.config.teacher_target_scope=='choice_or_all' and 'Choices:' in working.question)
        teacher_version=teacher.probabilities._version
        began=time.perf_counter()
        if usable:
            memory.native_mode=False
            params=(memory.gain,memory.value_coefficients)
            optimizer=torch.optim.Adam([
                dict(params=[params[0]],lr=engine.config.gain_learning_rate),
                dict(params=[params[1]],lr=engine.config.value_learning_rate)],foreach=False)
            try:
                max_steps=engine.config.max_answer_steps if use_choice else engine.config.steps
                for index in range(max_steps):
                    engine.budget.check();optimizer.zero_grad(set_to_none=True)
                    # Refresh from the current student on every update. The teacher
                    # then scores exactly these prefixes, never its own answer.
                    rollout=student_rollout(engine,reader,cache,suffix,
                        min(working.max_new_tokens,engine.config.student_rollout_tokens))
                    teacher_probs=score_student_prefix(engine,teacher_reader,teacher,rollout)
                    positions=choice_positions(engine.tokenizer,rollout.tolist()) if use_choice else []
                    target_positions=(torch.tensor(positions,device=rollout.device,dtype=torch.long)
                                      if positions else None)
                    target=(teacher_probs.index_select(0,target_positions) if positions else teacher_probs)
                    with torch.enable_grad():
                        logits=batched_teacher_logits(engine,reader,cache,suffix,rollout,
                                                      target_positions=target_positions)
                        loss=distribution_kl(logits,target)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite teacher KL')
                        before=float(loss.detach())
                        if index>=engine.config.steps and (not positions or before<=engine.config.answer_KL_stop):
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
                    steps.append(dict(step=index+1,accepted=True,teacher_KL_before=before,
                        student_rollout_tokens=rollout.tolist(),
                        target_positions=positions or list(range(len(rollout))),
                        gain_gradient_norm=norms[0],value_gradient_norm=norms[1],gradient_norm=total))
                if final_kl is None:
                    with torch.no_grad():
                        final_kl=float(distribution_kl(
                            batched_teacher_logits(engine,reader,cache,suffix,rollout,
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
            output = private_forward(reader.reader, suffix, length, branch)
            generated = [int(engine.model.lm_head(output.last_hidden_state[0, -1:])[0].argmax())]
            engine.synchronize()
            ttft = time.perf_counter() - started
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
                adaptive_max_answer_steps=engine.config.max_answer_steps,
                adaptive_answer_KL_stop=engine.config.answer_KL_stop,
                adapted_layers=memory.adapted_layers,
                checkpointed_layers=True,causal_batched_teacher_forcing=True,
            ),
            timing=dict(
                ttft_seconds=ttft,
                prefill_seconds=prefill_seconds,
                teacher_seconds=teacher.elapsed_seconds,
                support_seconds=0.0,adapter_setup_seconds=setup_seconds,
                adaptation_seconds=adaptation_seconds,
                total_seconds=time.perf_counter() - started,
                selection_backend=prepared.metadata.get("backend", "online_decisions"),
                offline_selection_seconds=prepared.metadata.get("offline_scoring_seconds", 0),
                offline_selection_cost_usd=prepared.metadata.get("offline_scoring_cost_usd", 0),
                offline_selection_in_ttft=False,
                scope=(
                    "Online request including selection JSON loading or live selection, teacher, "
                    "adaptation and fresh question through the first final token; excludes model loading, "
                    "queue time, and previously completed shared offline selection"
                ),
            ),
            backbone_unchanged=True,
            original_document_KV_unchanged=True,
            final_question_recomputed=True,
            teacher_answer_in_final_prompt=False,
        )
