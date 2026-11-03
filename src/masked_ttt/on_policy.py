"""Fresh autoregressive student rollouts for request-local distillation."""

import torch

from .runtime import private_forward


@torch.no_grad()
def student_rollout(engine, reader, cache, suffix, limit):
    """Sample the current student's greedy trajectory without retaining suffix KV."""
    if suffix.ndim != 1 or not len(suffix) or limit < 1:
        raise ValueError("A nonempty question and positive rollout budget are required")
    branch = reader.compiled_branch(cache)
    engine.budget.check()
    output = private_forward(reader.reader, suffix, reader.memory.document_length, branch)
    tokens = []
    while len(tokens) < limit:
        token = int(engine.model.lm_head(output.last_hidden_state[0, -1:])[0].argmax())
        tokens.append(token)
        if token in engine.eos_ids or len(tokens) == limit:
            break
        engine.budget.check()
        output = private_forward(
            reader.reader,
            torch.tensor([token], device=engine.device),
            reader.memory.document_length + len(suffix) + len(tokens) - 1,
            branch,
        )
    return torch.tensor(tokens, dtype=torch.long, device=engine.device)
