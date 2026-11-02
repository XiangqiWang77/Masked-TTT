"""Shared frozen parameters, private module containers, and explicit token positions."""

import copy
from collections import OrderedDict
import torch


def private_module_tree(module):
    """Copy module containers/hooks, sharing only immutable parameters/buffers."""
    private = copy.copy(module)
    private._modules = {
        name: private_module_tree(child) if child is not None else None
        for name, child in module._modules.items()
    }
    private._parameters = module._parameters.copy()
    private._buffers = module._buffers.copy()
    # Foreground capture hooks must never run on the replay thread.
    for name in (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks_always_called",
    ):
        if hasattr(private, name):
            setattr(private, name, OrderedDict())
    return private


def private_forward(model, tokens, start, cache):
    if model.model.config.model_type in ("olmo3", "gpt_oss"):
        from .architectures import hybrid_forward

        return hybrid_forward(model.model, tokens, start, cache)
    positions = torch.arange(start, start + tokens.numel(), device=tokens.device)
    # Auxiliary memory has physical cache slots, but does not add text tokens
    # to the request. Preserve the original Question/generation RoPE positions.
    extra = getattr(cache, "_auxiliary_memory_tokens", 0)
    cache_positions = positions if not extra else positions + extra
    return model.model(
        input_ids=tokens[None],
        position_ids=positions[None],
        cache_position=cache_positions,
        attention_mask=torch.ones(
            1, start + extra + tokens.numel(), device=tokens.device, dtype=torch.long
        ),
        past_key_values=cache,
        use_cache=True,
    )
