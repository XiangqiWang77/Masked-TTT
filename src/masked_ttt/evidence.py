"""Request-local selected evidence planning for the Teacher and cartridge."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

from . import _layout
from .canonical_evidence import CanonicalEvidence, build_canonical_evidence


_CURRENT = ContextVar("masked_ttt_selected_evidence", default=None)


@contextmanager
def selected_evidence_mode(tokenizer, mode, *, sink_tokens):
    """Bind one request's optional text retokenization without global hooks."""
    if mode not in ("original", "canonical"):
        raise ValueError("Unknown selected evidence tokenization")
    value = (tokenizer, sink_tokens) if mode == "canonical" else None
    token = _CURRENT.set(value)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def selected_chunk_positions(layout, selected, plan, separator_length):
    """Map each original selected chunk to its selected-only KV positions."""
    if separator_length < 1:
        raise ValueError("A nonempty separator is required")
    if isinstance(plan, CanonicalEvidence):
        if tuple(selected) != plan.original_selected_chunks:
            raise ValueError("The canonical evidence and selected chunks differ")
        return [list(chunk) for chunk in plan.chunks]
    mapping, cursor, previous = {}, layout.initial, layout.initial - 1
    for original in plan.original_document_positions:
        if original != previous + 1:
            cursor += separator_length
        mapping[original] = cursor
        cursor += 1
        previous = original
    if cursor != plan.document_end:
        raise ValueError("Selected cache position mapping is inconsistent")
    try:
        return [[mapping[position] for position in layout.chunks[i]] for i in selected]
    except KeyError as exc:
        raise ValueError("Selected cache is missing a selected token") from exc


def selected_read_plan(ids, chunks, initial, selected, separator_ids):
    """Keep raw token IDs by default; optionally retokenize the selected text."""
    raw = _layout.selected_read_plan(ids, chunks, initial, selected, separator_ids)
    setting = _CURRENT.get()
    if setting is None:
        return raw
    tokenizer, sink_tokens = setting
    layout = SimpleNamespace(ids=ids, chunks=chunks, initial=initial)
    mapped = selected_chunk_positions(layout, selected, raw, len(separator_ids))
    return build_canonical_evidence(tokenizer, layout, selected, raw, mapped, sink_tokens=sink_tokens)
