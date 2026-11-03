"""One fresh selected-context teacher, with a fixed full-vocabulary target."""

from dataclasses import dataclass
import time

import torch

from ._layout import prefill_batches, selected_read_plan
from .cache import AppendOnlyDocumentCache, fork_prefix
from .runtime import private_forward
from .request import Request,prepare_layout


@dataclass
class TeacherSignal:
    tokens: torch.Tensor
    probabilities: torch.Tensor
    prediction: str
    complete: bool
    elapsed_seconds: float
    first_token_seconds: float
    prompt_tokens: int
    instruction_applied: bool = False
    document_cache: object = None
    question_tokens: torch.Tensor = None
    document_end: int = 0
    document_ids: tuple = ()

    def metadata(self):
        return dict(
            prediction=self.prediction,
            tokens=self.tokens.tolist(),
            complete=self.complete,
            prompt_tokens=self.prompt_tokens,
            elapsed_seconds=self.elapsed_seconds,
            first_token_seconds=self.first_token_seconds,
            first_token_scope='selected_prompt_planning_prefill_first_token',
            target_fixed=True,
            fresh_selected_context=True,
            response_instruction_applied=self.instruction_applied,
        )


@torch.no_grad()
def prefill_document(backbone, reader, ids, initial, boundaries):
    cache = AppendOnlyDocumentCache(backbone.model.config, boundaries[-1])
    tokens = torch.tensor(ids, device=backbone.device)
    start = 0
    for end, _ in prefill_batches(initial, boundaries, backbone.config.prefill_chunk_tokens):
        backbone.budget.check()
        private_forward(reader, tokens[start:end], start, cache)
        start = end
    backbone.synchronize()
    return cache


@torch.no_grad()
def build_teacher(backbone, reader, request, layout, selected):
    began = time.perf_counter()
    instruction=backbone.config.teacher_response_instruction
    teacher_ids=layout.ids
    if instruction:
        teacher_request=Request(document=request.document,question=request.question,
            prompt=request.prompt+'\n\nResponse requirements:\n'+instruction,
            answer_prefix=request.answer_prefix,max_new_tokens=request.max_new_tokens)
        teacher_layout=prepare_layout(backbone,teacher_request)
        if teacher_layout.ids[:layout.document_end]!=layout.ids[:layout.document_end]:
            raise ValueError('Teacher instruction changed the document token prefix')
        teacher_ids=teacher_layout.ids
    plan = selected_read_plan(
        teacher_ids,
        layout.chunks,
        layout.initial,
        selected,
        backbone.tokenizer.encode("\n\n", add_special_tokens=False),
    )
    cap = min(request.max_new_tokens, backbone.config.teacher_max_new_tokens)
    if len(plan.ids) + cap > backbone.config.max_context:
        raise ValueError("Selected-context teacher exceeds the model context")
    cache = prefill_document(
        backbone, reader, plan.ids, plan.initial_prefix_tokens, plan.chunk_boundaries
    )
    branch = fork_prefix(cache, plan.document_end)
    suffix = torch.tensor(plan.ids[plan.document_end :], device=backbone.device)
    output = private_forward(reader, suffix, plan.document_end, branch)
    trace, tokens, first = [], [], None
    while len(tokens) < cap:
        logits = backbone.model.lm_head(output.last_hidden_state[0, -1:])[0].float()
        trace.append(logits.softmax(-1).detach())
        tokens.append(int(logits.argmax()))
        if first is None:
            backbone.synchronize()
            first = time.perf_counter() - began
        if tokens[-1] in backbone.eos_ids or len(tokens) == cap:
            break
        backbone.budget.check()
        output = private_forward(
            reader,
            torch.tensor([tokens[-1]], device=backbone.device),
            len(plan.ids) + len(tokens) - 1,
            branch,
        )
    backbone.synchronize()
    return TeacherSignal(
        torch.tensor(tokens, device=backbone.device),
        torch.stack(trace),
        backbone.tokenizer.decode(tokens, skip_special_tokens=True),
        tokens[-1] in backbone.eos_ids,
        time.perf_counter() - began,
        first,
        len(plan.ids),
        bool(instruction),
        cache,
        suffix,
        plan.document_end,
        tuple(plan.ids[:plan.document_end]),
    )


@torch.no_grad()
def score_student_prefix(backbone, reader, teacher, student_tokens):
    """Teacher probabilities at the student's actual autoregressive prefixes.

    Reuse the selected-only document cache, but give the teacher exactly the
    same generated prefix the student saw. The current token is never input.
    """
    if student_tokens.ndim != 1 or not len(student_tokens):
        raise ValueError("A nonempty one-dimensional student trajectory is required")
    if teacher.document_cache is None or teacher.question_tokens is None:
        raise ValueError("The teacher must retain its selected-document prefix")
    tokens = torch.cat((teacher.question_tokens, student_tokens[:-1]))
    branch = fork_prefix(teacher.document_cache, teacher.document_end)
    backbone.budget.check()
    output = private_forward(reader, tokens, teacher.document_end, branch)
    hidden = output.last_hidden_state[0, len(teacher.question_tokens) - 1 :]
    return backbone.model.lm_head(hidden).float().softmax(-1).detach()
