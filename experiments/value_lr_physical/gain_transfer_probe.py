"""Read-only diagnostics for portability of per-chunk attention updates.

Run this in an isolated, single-request process. The observer records scalar
statistics only; it never supplies a signal to the optimizer or the answer.
"""

from contextlib import contextmanager
import math
import weakref

import torch


def chunk_components(values):
    """Project [layer, head, selected chunk] values into two orthogonal parts.

    The common component is constant across selected chunks within each head;
    the contrast is the remaining chunk-dependent component. Equal changes to
    all selected-document scores cancel only in a document-normalized softmax:
    in an actual model they can still change attention to sinks and suffixes.
    """
    if values.ndim != 3 or values.shape[-1] == 0:
        raise ValueError("Expected layers, heads and at least one selected chunk")
    sample = values.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(sample).all()):
        raise ValueError("Non-finite gain signal")
    common = sample.mean(dim=-1, keepdim=True)
    common_sq = float((common.square().sum() * sample.shape[-1]).item())
    contrast_sq = float((sample - common).square().sum().item())
    total_sq = common_sq + contrast_sq
    return dict(
        selected_chunks=sample.shape[-1],
        common_norm=math.sqrt(common_sq),
        contrast_norm=math.sqrt(contrast_sq),
        total_norm=math.sqrt(total_sq),
        common_squared_fraction=common_sq / total_sq if total_sq > 0 else None,
    )


class _GainRecorder:
    def __init__(self):
        self.steps = []
        self.pending = None

    def gradient(self, memory_ref, gradient):
        memory = memory_ref()
        if memory is None:
            raise RuntimeError("The trainable memory disappeared during backward")
        if self.pending is not None:
            raise RuntimeError("A gain update is already pending")
        with torch.no_grad():
            # A selected chunk containing only excluded sink tokens has no
            # addressable positions and does not participate in the projection.
            chunks = memory.chunk_index[memory.selected_positions].unique(sorted=True)
            layers = torch.as_tensor(memory.adapted_layers, device=gradient.device)
            if not len(chunks) or not len(layers):
                self.pending = dict(unavailable="no_active_selected_positions")
                return
            chunks = chunks.to(device=gradient.device)
            grad = gradient.index_select(0, layers).index_select(-1, chunks)
            before = memory.gain.detach().index_select(0, layers).index_select(-1, chunks).clone()
            self.pending = dict(layers=layers, chunks=chunks, before=before,
                                gradient=chunk_components(grad))
        # Returning None leaves the gradient untouched.

    def constrained(self, memory):
        if self.pending is None:
            raise RuntimeError("A gain update occurred without a recorded gradient")
        pending, self.pending = self.pending, None
        if "unavailable" in pending:
            self.steps.append(pending)
            return
        with torch.no_grad():
            after = memory.gain.detach().index_select(0, pending["layers"]).index_select(-1, pending["chunks"])
            self.steps.append(dict(
                gradient=pending["gradient"],
                optimizer_update=chunk_components(after - pending["before"]),
                final_gain=chunk_components(after),
            ))


@contextmanager
def capture_gain_transfer():
    """Observe the existing full-student training; preserve its parameters.

    This patches only the pipeline's private-memory constructor and is not a
    concurrent-request API. Hooks use weak references so the observer cannot
    retain full or selected KV, or the training memory, at final decoding.
    """
    import masked_ttt.pipeline as pipeline

    original = pipeline.KVReadMemory
    recorder = _GainRecorder()

    class ObservedMemory(original):
        def to(self, *args, **kwargs):
            result = super().to(*args, **kwargs)
            if not getattr(result, "_gain_probe_attached", False):
                memory_ref = weakref.ref(result)
                result.gain.register_hook(lambda gradient: recorder.gradient(memory_ref, gradient))
                result._gain_probe_attached = True
            return result

        def constrain(self):
            result = super().constrain()
            recorder.constrained(self)
            return result

    pipeline.KVReadMemory = ObservedMemory
    try:
        yield recorder.steps
        if recorder.pending is not None:
            raise RuntimeError("A recorded gradient was not followed by an update")
    finally:
        pipeline.KVReadMemory = original
        recorder.pending = None
