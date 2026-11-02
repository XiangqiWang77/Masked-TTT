"""One fresh selected-context teacher, with a fixed full-vocabulary target."""

from dataclasses import dataclass
import hashlib
import time

import torch

from ._layout import prefill_batches, selected_read_plan
from .cache import AppendOnlyDocumentCache, fork_prefix
from .runtime import private_forward


@dataclass
class TeacherSignal:
    tokens: torch.Tensor
    probabilities: torch.Tensor
    prediction: str
    complete: bool
    elapsed_seconds: float
    first_token_seconds: float
    prompt_tokens: int

    def metadata(self):
        return dict(
            prediction=self.prediction,
            tokens=self.tokens.tolist(),
            complete=self.complete,
            prompt_tokens=self.prompt_tokens,
            elapsed_seconds=self.elapsed_seconds,
            first_token_seconds=self.first_token_seconds,
            distribution_sha256=hashlib.sha256(
                self.probabilities.contiguous().cpu().numpy().tobytes()
            ).hexdigest(),
            target_fixed=True,
            fresh_selected_context=True,
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
    plan = selected_read_plan(
        layout.ids,
        layout.chunks,
        layout.initial,
        selected,
        backbone.tokenizer.encode("\n\n", add_special_tokens=False),
    )
    cap = min(request.max_new_tokens, backbone.config.teacher_max_new_tokens)
    if len(plan.ids) + cap > backbone.config.max_context:
        raise ValueError("Selected-context teacher exceeds the model context")
    began = time.perf_counter()
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
    )
