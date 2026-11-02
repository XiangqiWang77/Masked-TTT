"""Exact prompt token boundaries, selected-only teacher plans, and prefill batching."""

from dataclasses import dataclass
import re


def chunk_positions(
    offsets, document_start, document_end, text, maximum=1024, boundary_mode="legacy"
):
    """Group source labels, preferentially ending at sentence/paragraph breaks.

    Offsets are for the COMPLETE original chat prompt. Position t uses feature t-1.
    Only tokens entirely inside the source are supervised, including its first
    token when a preceding prompt token exists. No prompt retokenization/splicing.
    """
    if maximum < 2:
        raise ValueError("Chunks require at least two probe targets")
    if boundary_mode not in ("legacy", "token_aligned"):
        raise ValueError("Unknown source boundary mode")
    positions = [
        i for i, (a, b) in enumerate(offsets) if i > 0 and document_start <= a < b <= document_end
    ]
    boundaries = {m.end() for m in re.finditer(r"\n\s*\n|[.!?。！？][ \t\n]+|\n", text)}
    if boundary_mode == "token_aligned":
        # Byte-level tokenizers often attach the following space to the NEXT
        # word. A regex boundary after that space lies inside that token, so
        # the legacy equality check misses an otherwise valid sentence end.
        boundaries.update(m.start() + 1 for m in re.finditer(r"[.!?。！？](?:\s|$)", text))
    result = []
    start = 0
    while start < len(positions):
        stop = min(start + maximum, len(positions))
        if stop < len(positions):
            candidates = [
                j
                for j in range(start + maximum // 2, stop + 1)
                if offsets[positions[j - 1]][1] in boundaries
                or (j < len(positions) and offsets[positions[j]][0] in boundaries)
            ]
            if candidates:
                stop = candidates[-1]
            elif boundary_mode == "token_aligned":
                # Prefer a word break to a hard cap through a multi-token
                # number/name. Keep the original token IDs and hard maximum;
                # genuinely overlong words still require the hard-cap split.
                candidates = []
                for j in range(start + maximum // 2, stop + 1):
                    left, right = offsets[positions[j - 1]][1], offsets[positions[j]][0]
                    if left <= right and (
                        text[left:right].isspace()
                        or (left > 0 and text[left - 1].isspace())
                        or (right < len(text) and text[right].isspace())
                    ):
                        candidates.append(j)
                if candidates:
                    stop = candidates[-1]
        result.append(positions[start:stop])
        start = stop
    return result


@dataclass(frozen=True)
class EvidenceReadPlan:
    ids: tuple
    document_end: int
    chunk_boundaries: tuple
    initial_prefix_tokens: int
    original_document_positions: tuple
    separators_added: int


def original_read_plan(ids, chunks, initial):
    _validate(ids, chunks, initial)
    return EvidenceReadPlan(
        tuple(ids),
        chunks[-1][-1] + 1,
        tuple(c[-1] + 1 for c in chunks),
        initial,
        tuple(range(initial, chunks[-1][-1] + 1)),
        0,
    )


def selected_read_plan(ids, chunks, initial, selected, separator_ids):
    _validate(ids, chunks, initial)
    if (
        not isinstance(selected, (list, tuple))
        or any(isinstance(i, bool) or not isinstance(i, int) for i in selected)
        or list(selected) != sorted(set(selected))
        or any(i < 0 or i >= len(chunks) for i in selected)
    ):
        raise ValueError("Selection must be unique chronological whole-chunk indices")
    if not separator_ids or any(
        isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in separator_ids
    ):
        raise ValueError("A real tokenizer separator is required")
    if list(selected) == list(range(len(chunks))):
        return original_read_plan(ids, chunks, initial)
    # A boundary-crossing BPE token can be read normally but be omitted from
    # the original training-target chunks. Preserve every such token: it can
    # contain a real first source character, not merely whitespace.
    document_positions = set(range(initial, chunks[-1][-1] + 1))
    assigned = {p for c in chunks for p in c}
    keep = (document_positions - assigned) | {p for i in selected for p in chunks[i]}
    ends = {chunks[i][-1] for i in selected}
    result = list(ids[:initial])
    positions = []
    boundaries = []
    previous = initial - 1
    separators = 0
    for position in sorted(keep):
        if position != previous + 1:
            result.extend(separator_ids)
            separators += 1
        result.append(ids[position])
        positions.append(position)
        previous = position
        if position in ends:
            boundaries.append(len(result))
    if len(result) > initial and (not boundaries or boundaries[-1] != len(result)):
        boundaries.append(len(result))
    stop = len(result)
    result.extend(ids[chunks[-1][-1] + 1 :])
    return EvidenceReadPlan(
        tuple(result), stop, tuple(boundaries), initial, tuple(positions), separators
    )


def _validate(ids, chunks, initial):
    if (
        not isinstance(initial, int)
        or isinstance(initial, bool)
        or initial < 1
        or not chunks
        or any(not c for c in chunks)
        or chunks[0][0] < initial
        or chunks[-1][-1] + 1 >= len(ids)
    ):
        raise ValueError("Original document, task prefix and Question suffix are required")
    positions = [p for c in chunks for p in c]
    if positions != sorted(set(positions)) or any(
        c != list(range(c[0], c[-1] + 1)) for c in chunks
    ):
        raise ValueError(
            "Whole source chunks must be chronological, disjoint and locally consecutive"
        )


def prefill_batches(initial_prefix, boundary_lengths, maximum_tokens=None):
    if initial_prefix < 1 or not boundary_lengths:
        raise ValueError("A task prefix and source boundaries are required")
    if any(a >= b for a, b in zip([initial_prefix] + list(boundary_lengths), boundary_lengths)):
        raise ValueError("Source boundaries must increase after the task prefix")
    if maximum_tokens is not None and (
        isinstance(maximum_tokens, bool)
        or not isinstance(maximum_tokens, int)
        or maximum_tokens < 1
    ):
        raise ValueError("Foreground batch size must be a positive integer or None")
    result = [(initial_prefix, [(-1, initial_prefix)])]
    start, pending = initial_prefix, []
    for chunk, end in enumerate(boundary_lengths):
        if pending and (maximum_tokens is None or end - start > maximum_tokens):
            result.append((pending[-1][1], pending))
            start, pending = pending[-1][1], []
        pending.append((chunk, end))
    if pending:
        result.append((pending[-1][1], pending))
    return result
