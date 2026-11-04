"""Opt-in canonical selected-text Teacher and cartridge, preserving original SD."""
from contextlib import contextmanager
from types import SimpleNamespace

from experiments.value_lr_physical.canonical_evidence import CanonicalEvidence, build_canonical_evidence


@contextmanager
def canonical_selected_evidence(tokenizer, *, sink_tokens):
    """Retokenize the selected document for Teacher and standalone prefill.

    The frozen full-document Student, on-policy objective, and chunk parameters
    are unchanged. Selected evidence is retokenized independently with the
    original prefix and suffix token IDs held fixed. This experimental hook is
    process-local and must not run concurrently with other requests.
    """
    from masked_ttt import _layout, self_study_runtime, shared_selected, standalone, teacher

    modules = (teacher, self_study_runtime, shared_selected, standalone)
    originals = [module.selected_read_plan for module in modules]
    old_standalone_map = standalone.selected_chunk_positions
    old_shared_map = shared_selected.selected_chunk_positions
    if any(fn is not _layout.selected_read_plan for fn in originals):
        raise ValueError("The selected-evidence planning path is already patched")
    if old_shared_map is not old_standalone_map:
        raise ValueError("The selected-evidence chunk mapping is already patched")

    def selected_plan(ids, chunks, initial, selected, separator_ids):
        raw = _layout.selected_read_plan(ids, chunks, initial, selected, separator_ids)
        layout = SimpleNamespace(ids=ids, chunks=chunks, initial=initial)
        mapped = old_standalone_map(layout, selected, raw, len(separator_ids))
        return build_canonical_evidence(tokenizer, layout, selected, raw, mapped, sink_tokens=sink_tokens)

    def selected_positions(_layout_value, selected, plan, _separator_length):
        if not isinstance(plan, CanonicalEvidence) or tuple(selected) != plan.original_selected_chunks:
            raise ValueError("The canonical evidence and selected chunks differ")
        return [list(chunk) for chunk in plan.chunks]

    for module in modules:
        module.selected_read_plan = selected_plan
    standalone.selected_chunk_positions = shared_selected.selected_chunk_positions = selected_positions
    try:
        yield
    finally:
        for module, original in zip(modules, originals):
            module.selected_read_plan = original
        standalone.selected_chunk_positions = old_standalone_map
        shared_selected.selected_chunk_positions = old_shared_map
