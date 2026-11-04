"""Keep only materialized selected KV and lightweight routing metadata.

This single-request experiment does not change training or the readout. Once
the frozen gain and value correction have been baked into the selected KV,
their duplicate inference-time tensors are no longer read by the augmented
attention path. The existing training state is released independently.
"""
from contextlib import contextmanager

from .memory_accounting import cache_tensors, module_tensors, storage_bytes
from .standalone import StandaloneReady


def _drop_redundant_tensors(ready):
    if not isinstance(ready, StandaloneReady):
        raise RuntimeError("Deferred standalone decoding is required")
    memory = ready.reader.memory
    if memory.capture_attention:
        raise ValueError("Attention capture requires the full reader metadata")
    if not memory.native_mode and not getattr(ready.branch, "_masked_ttt_augmented", False):
        raise ValueError("An active update must already be materialized into the cache")
    if not memory.native_mode:
        padded = ((memory.dim + 1 + 31) // 32) * 32
        for index in memory.adapted_layers:
            layer = ready.branch.layers[index]
            if (layer.keys is None or layer.values is None or
                layer.keys.shape[-1] != padded or layer.values.shape[-1] != padded):
                raise ValueError("An adapted layer does not contain the materialized KV")
    if not ready.release_source or not ready.release_training:
        raise ValueError("The selected source and training state must both be released")

    before = storage_bytes(module_tensors(memory))
    # Attention closures retain the same module object. In the materialized
    # branch they consult only scalar flags and adapted layer IDs; neither
    # stored parameter nor selection-index buffer is used for decoding.
    for name in tuple(memory._parameters):
        delattr(memory, name)
    for name in tuple(memory._buffers):
        delattr(memory, name)
    if module_tensors(memory):
        raise AssertionError("The inference reader still owns adapter tensors")

    cache_bytes = storage_bytes(cache_tensors(ready.branch))
    ready.accounting.update(
        cartridge_adapter_bytes=0,
        standalone_cartridge_bytes=cache_bytes,
        kv_only_adapter_bytes_removed=before,
        kv_only_materialized=True,
        kv_only_native_mode=bool(memory.native_mode),
    )
    return before


@contextmanager
def kv_only_cartridge():
    """Remove the duplicate adapter immediately before independent decoding.

    Hooks are process-local and must not be used for concurrent requests.
    The selected KV and all model weights retain their original storage.
    """
    import masked_ttt.pipeline as pipeline

    original = pipeline.answer_standalone

    def prepare(engine, layout, selected, teacher, trained, limit):
        ready = original(engine, layout, selected, teacher, trained, limit)
        _drop_redundant_tensors(ready)
        return ready

    pipeline.answer_standalone = prepare
    try:
        yield
    finally:
        pipeline.answer_standalone = original
