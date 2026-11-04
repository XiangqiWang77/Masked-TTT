"""Build a candidate canonical evidence-token plan with explicit chunk ownership.

This opt-in plan leaves the full-document Student untouched. The selected
Teacher and standalone cartridge share its independently prefetched evidence.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalEvidence:
    ids: tuple
    initial_prefix_tokens: int
    document_end: int
    prefill_boundaries: tuple
    chunks: tuple
    original_selected_chunks: tuple
    mixed_boundary_tokens: int

    @property
    def chunk_boundaries(self):
        return self.prefill_boundaries


def build_canonical_evidence(tokenizer, layout, selected, plan, mapped, *, sink_tokens):
    """Retokenize selected document text; reject ambiguous or lossy ownership.

    A token overlapping one selected chunk and a separator belongs to that
    chunk. A token overlapping two selected chunks is rejected because one
    token cannot carry two independent chunk corrections. Prefix and question
    suffix IDs are kept unchanged; this is not whole-prompt retokenization.
    """
    if (not selected or list(selected) != sorted(set(selected)) or len(mapped) != len(selected) or
        type(sink_tokens) is not int or sink_tokens < 0 or layout.initial < 1 or
        any(not positions or positions != list(range(positions[0], positions[-1] + 1)) for positions in mapped)):
        raise ValueError("Expected chronological selected chunks with consecutive evidence-token positions")
    raw = list(plan.ids[layout.initial:plan.document_end])
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = decode(raw)
    if not text or "\ufffd" in text:
        raise ValueError("Evidence decoding is empty or contains a replacement character")
    ranges = []
    for positions in mapped:
        left, right = positions[0] - layout.initial, positions[-1] + 1 - layout.initial
        if not 0 <= left < right <= len(raw):
            raise ValueError("Mapped chunk is outside the evidence document")
        before, through = decode(raw[:left]), decode(raw[:right])
        if not text.startswith(before) or not text.startswith(through) or len(through) <= len(before):
            raise ValueError("A selected chunk boundary splits a decoded character")
        ranges.append((len(before), len(through)))
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Selected chunk character ranges overlap")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    canonical, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if (len(canonical) != len(offsets) or
        canonical != tokenizer.encode(text, add_special_tokens=False) or decode(canonical) != text):
        raise ValueError("Canonical evidence tokenization did not round-trip")
    coverage = [bytearray(right-left) for left, right in ranges]
    chunks = [[] for _ in ranges]
    mixed = 0
    for local_position, (start, end) in enumerate(offsets):
        if not 0 <= start <= end <= len(text):
            raise ValueError("Canonical token offset is outside the evidence text")
        owners = [i for i, (left, right) in enumerate(ranges) if max(left, start) < min(right, end)]
        if len(owners) > 1:
            raise ValueError("A canonical token overlaps multiple selected chunks")
        if not owners:
            continue
        owner = owners[0]
        left, right = ranges[owner]
        position = layout.initial + local_position
        if position < sink_tokens:
            raise ValueError("A canonical selected token overlaps the attention sink")
        chunks[owner].append(position)
        mixed += int(start < left or end > right)
        for char in range(max(start, left), min(end, right)):
            coverage[owner][char-left] = 1
    if any(not positions or positions != list(range(positions[0], positions[-1] + 1)) for positions in chunks):
        raise ValueError("A selected chunk has empty or discontinuous canonical ownership")
    if any(not covered and not text[left + index].isspace()
           for (left, _), mask in zip(ranges, coverage) for index, covered in enumerate(mask)):
        raise ValueError("Canonical token offsets omit a non-whitespace evidence character")
    document_end = layout.initial + len(canonical)
    boundaries = sorted(set([chunk[-1] + 1 for chunk in chunks] + [document_end]))
    ids = tuple(plan.ids[:layout.initial]) + tuple(canonical) + tuple(plan.ids[plan.document_end:])
    return CanonicalEvidence(ids=ids, initial_prefix_tokens=layout.initial, document_end=document_end,
        prefill_boundaries=tuple(boundaries), chunks=tuple(tuple(chunk) for chunk in chunks),
        original_selected_chunks=tuple(selected), mixed_boundary_tokens=mixed)
