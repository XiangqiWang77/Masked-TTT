"""Select -> frozen teacher -> source support -> joint KV SD -> fresh answer."""

from copy import copy
from types import SimpleNamespace
import time

import torch

from .cache import fork_prefix
from .checkpointing import layer_checkpointed_logits
from .memory import KVReadMemory
from .model import Budget
from .optimizer import distribution_kl, fit_memory
from .query_guard import question_token_sets
from .reader import MemoryReader, matched_incremental_logits
from .runtime import private_forward, private_module_tree
from .selection import OpenAIChunkSelector
from .support import check_signal
from .teacher import build_teacher, prefill_document


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
        with torch.no_grad():
            # Preserve the submitted initialization: native question read, then
            # a fixed source-value eigenspace in every adapted layer.
            memory.native_mode, memory.capture_attention = True, True
            private_forward(reader.reader, suffix, length, fork_prefix(cache, length))
            memory.capture_attention = False
            layers = sorted(
                memory.adapted_layers,
                key=lambda layer: (-memory.attention_records[layer]["selected_mass"], layer),
            )
            memory.prepare_value_basis(cache, layers)
            memory.attention_records = {}

        checkpointed = len(
            layout.ids
        ) >= engine.config.checkpoint_at_tokens or engine.model.config.model_type in (
            "olmo3",
            "gpt_oss",
        )

        def evaluate():
            if torch.is_grad_enabled():
                memory.native_mode = False
                if checkpointed:
                    return layer_checkpointed_logits(engine, reader, cache, suffix, teacher.tokens)
            else:
                memory.select_actual_read_path()
            return matched_incremental_logits(engine, reader, cache, suffix, teacher.tokens)

        with torch.no_grad():
            initial_kl = float(distribution_kl(evaluate(), teacher.probabilities))
            before_check = memory.digest()
            began = time.perf_counter()
            support = check_signal(
                engine, teacher_reader, cache, ids, working, layout, selected, teacher
            )
            support_seconds = time.perf_counter() - began
            if memory.digest() != before_check:
                raise AssertionError("Source checking changed the adaptation state")

        content, qualifiers = question_token_sets(engine.tokenizer, working.question)
        began = time.perf_counter()
        with torch.enable_grad():
            steps, final_kl, diagnostics = fit_memory(
                memory,
                evaluate,
                teacher.probabilities,
                support["decision"]["planned_steps"],
                engine.config.update,
                engine.budget.check,
                query_token_ids=content,
                query_qualifier_ids=qualifiers,
            )
        engine.synchronize()
        adaptation_seconds = time.perf_counter() - began
        if not support["decision"]["apply_update"] and (
            bool(memory.gain.any()) or bool(memory.value_coefficients.any())
        ):
            raise AssertionError("An unadmitted teacher signal changed memory")

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
                objective="KL(teacher || student)",
                initial_kl=initial_kl,
                final_kl=final_kl,
                steps=steps,
                accepted_steps=sum(r["accepted"] for r in steps),
                diagnostics=diagnostics,
                adapted_layers=memory.adapted_layers,
                checkpointed_layers=checkpointed,
            ),
            timing=dict(
                ttft_seconds=ttft,
                prefill_seconds=prefill_seconds,
                teacher_seconds=teacher.elapsed_seconds,
                support_seconds=support_seconds,
                adaptation_seconds=adaptation_seconds,
                total_seconds=time.perf_counter() - started,
                selection_backend=prepared.metadata.get("backend", "online_decisions"),
                offline_selection_seconds=prepared.metadata.get("offline_scoring_seconds", 0),
                offline_selection_cost_usd=prepared.metadata.get("offline_scoring_cost_usd", 0),
                offline_selection_in_ttft=False,
                scope=(
                    "Online request including selection JSON loading or live selection, teacher, checks, "
                    "adaptation and fresh question through the first final token; excludes model loading, "
                    "queue time, and previously completed shared offline selection"
                ),
            ),
            backbone_unchanged=True,
            original_document_KV_unchanged=True,
            final_question_recomputed=True,
            teacher_answer_in_final_prompt=False,
        )
