"""Immutable document KV storage and private append-only suffix branches."""

import copy
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class AppendOnlyDocumentCache(DynamicCache):
    def __init__(self, config, capacity):
        super().__init__(config=config)
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("Document cache capacity must be a positive integer")
        if any((type(layer) is not DynamicLayer for layer in self.layers)):
            raise ValueError("Append-only document cache requires full attention layers")
        self.capacity = capacity
        self.storage = [None] * len(self.layers)
        self._source_identity_token = object()

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if torch.is_grad_enabled() or key_states.requires_grad or value_states.requires_grad:
            raise ValueError("Only frozen foreground document states may be appended")
        layer = self.layers[layer_idx]
        (start, size) = (layer.get_seq_length(), key_states.shape[-2])
        if key_states.ndim != 4 or key_states.shape != value_states.shape or size < 1:
            raise ValueError("Invalid document key/value shape")
        end = start + size
        if end > self.capacity:
            raise ValueError("Document cache capacity exceeded; published slots cannot be reused")
        if self.storage[layer_idx] is None:
            shape = (*key_states.shape[:2], self.capacity, key_states.shape[-1])
            keys = torch.empty(shape, dtype=key_states.dtype, device=key_states.device)
            values = torch.empty_like(keys)
            self.storage[layer_idx] = (keys, values)
            (layer.dtype, layer.device, layer.is_initialized) = (keys.dtype, keys.device, True)
        (keys, values) = self.storage[layer_idx]
        if (
            key_states.shape[:2] != keys.shape[:2]
            or key_states.shape[-1] != keys.shape[-1]
            or key_states.dtype != keys.dtype
            or (value_states.dtype != values.dtype)
            or (key_states.device != keys.device)
            or (value_states.device != values.device)
        ):
            raise ValueError("Document KV layout changed within a request")
        keys[..., start:end, :].copy_(key_states)
        values[..., start:end, :].copy_(value_states)
        (layer.keys, layer.values) = (keys[..., :end, :], values[..., :end, :])
        return (layer.keys, layer.values)

    def snapshot(self, length):
        if not 0 < length <= self.capacity:
            raise ValueError("Invalid document publication length")
        if any((layer.get_seq_length() < length for layer in self.layers)):
            raise ValueError("Cannot publish a prefix before every layer has written it")
        branch = DynamicCache()
        branch.layers = []
        for layer in self.layers:
            item = copy.copy(layer)
            (item.keys, item.values) = (layer.keys[..., :length, :], layer.values[..., :length, :])
            branch.layers.append(item)
        branch._document_storage_owner = self
        branch._source_identity_token = self._source_identity_token
        return branch

    def crop(self, *args, **kwargs):
        raise RuntimeError("Published document slots may never be cropped or overwritten")

    def reset(self, *args, **kwargs):
        raise RuntimeError("Discard the request cache instead of reusing published slots")

    def batch_repeat_interleave(self, *args, **kwargs):
        raise RuntimeError("The document cache layout is fixed for one request")

    def batch_select_indices(self, *args, **kwargs):
        raise RuntimeError("The document cache layout is fixed for one request")


def fork_prefix(cache, length):
    """Share immutable prefix tensors; append only in independent layer objects."""
    if isinstance(cache, AppendOnlyDocumentCache):
        return cache.snapshot(length)
    if not isinstance(cache, DynamicCache) or cache.offloading:
        raise TypeError("Prefix reuse requires a non-offloaded DynamicCache")
    branch = copy.copy(cache)
    branch.layers = []
    for layer in cache.layers:
        # DynamicLayer.update uses out-of-place torch.cat, not indexed writes.
        if type(layer) is not DynamicLayer or layer.get_seq_length() < length:
            raise TypeError("Only full-attention DynamicLayer prefixes are supported")
        item = copy.copy(layer)
        item.keys = layer.keys[..., :length, :]
        item.values = layer.values[..., :length, :]
        branch.layers.append(item)
    return branch
