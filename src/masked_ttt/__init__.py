"""Masked TTT; CPU selection utilities do not import model-compute libraries."""

from importlib import import_module

_EXPORTS = {
    "DecisionConfig": "config",
    "TTTConfig": "config",
    "Backbone": "model",
    "MaskedTTT": "pipeline",
    "Request": "request",
    "OpenAIChunkSelector": "selection",
    "FrozenJSONSelector": "frozen_scores",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Backbone",
    "DecisionConfig",
    "MaskedTTT",
    "OpenAIChunkSelector",
    "FrozenJSONSelector",
    "Request",
    "TTTConfig",
]
