from types import SimpleNamespace
import random

import pytest

from experiments.value_lr_physical.audit_selected_tokenization import (FinalChannelTokenizer,
    inspect_plan, selected_chunk_positions, summarize)
from experiments.value_lr_physical.canonical_evidence import build_canonical_evidence
from masked_ttt._layout import selected_read_plan
from masked_ttt.architectures import FinalChannelTokenizer as ProductionFinalChannelTokenizer
from masked_ttt.standalone import selected_chunk_positions as production_mapping


class CharTokenizer:
    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **_kwargs):
        return "".join(chr(value) for value in ids)

    def __call__(self, text, **_kwargs):
        return dict(input_ids=self.encode(text), offset_mapping=[(i, i+1) for i in range(len(text))])


def test_selected_mapping_and_new_sink_positions_are_counted():
    tokenizer = CharTokenizer()
    ids = tokenizer.encode("XYabcdefghZ")
    layout = SimpleNamespace(ids=ids, initial=2, chunks=[[2, 3, 4, 5], [6, 7], [8, 9]])
    row = inspect_plan(tokenizer, layout, [1], sink_tokens=5)
    assert row["exact_selected_token_mapping"] and row["selected_source_tokens"] == 2
    assert row["sink_membership_changed"] == 1
    assert row["canonical_identical"] and row["replacement_characters"] == 0
    totals = summarize([row])
    assert totals["cases"] == totals["exact_selected_token_mapping"] == 1
    assert totals["sink_membership_changed_cases"] == 1


def test_noncanonical_ids_do_not_imply_changed_decoded_text():
    class MergeTokenizer(CharTokenizer):
        def encode(self, text, **_kwargs):
            if text == "ab":
                return [1000]
            return super().encode(text)
        def decode(self, ids, **_kwargs):
            return "".join("ab" if value == 1000 else chr(value) for value in ids)
        def __call__(self, text, **_kwargs):
            if text == "ab":
                return dict(input_ids=[1000], offset_mapping=[(0, 2)])
            return super().__call__(text)
    tokenizer = MergeTokenizer()
    layout = SimpleNamespace(ids=[88, 89, 97, 98, 99, 100, 90], initial=2,
                             chunks=[[2, 3], [4, 5]])
    row = inspect_plan(tokenizer, layout, [0], sink_tokens=2)
    assert not row["canonical_identical"] and row["canonical_length_delta"] == -1
    assert row["exact_selected_token_mapping"] and row["replacement_characters"] == 0
    assert row["canonical_ownership"]["ambiguous_chunk_tokens"] == 0


def test_canonical_token_may_include_separator_and_one_chunk():
    class JoinedTokenizer(CharTokenizer):
        def encode(self, text, **_kwargs):
            if text == "ab\n\nEF":
                return [97, 98, 1000, 70]
            return super().encode(text)
        def decode(self, ids, **_kwargs):
            return "".join("\n\nE" if value == 1000 else chr(value) for value in ids)
        def __call__(self, text, **_kwargs):
            if text == "ab\n\nEF":
                return dict(input_ids=[97, 98, 1000, 70], offset_mapping=[(0, 1), (1, 2), (2, 5), (5, 6)])
            return super().__call__(text)
    tokenizer = JoinedTokenizer()
    layout = SimpleNamespace(ids=CharTokenizer().encode("XYabCDEFZ"), initial=2,
                             chunks=[[2, 3], [4, 5], [6, 7]])
    ownership = inspect_plan(tokenizer, layout, [0, 2], sink_tokens=2)["canonical_ownership"]
    assert ownership["stable_chunk_boundaries"] and ownership["mixed_separator_tokens"] == 1
    assert ownership["ambiguous_chunk_tokens"] == ownership["uncovered_nonspace_characters"] == 0
    plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, [0, 2], [10, 10])
    mapped = selected_chunk_positions(layout, [0, 2], plan, 2)
    candidate = build_canonical_evidence(tokenizer, layout, [0, 2], plan, mapped, sink_tokens=2)
    assert candidate.ids == (88, 89, 97, 98, 1000, 70, 90)
    assert candidate.chunks == ((2, 3), (4, 5)) and candidate.prefill_boundaries == (4, 6)
    assert candidate.mixed_boundary_tokens == 1 and candidate.original_selected_chunks == (0, 2)


def test_canonical_token_spanning_two_chunks_is_reported_ambiguous():
    class CrossTokenizer(CharTokenizer):
        def encode(self, text, **_kwargs):
            if text == "abCD":
                return [97, 1000, 68]
            return super().encode(text)
        def decode(self, ids, **_kwargs):
            return "".join("bC" if value == 1000 else chr(value) for value in ids)
        def __call__(self, text, **_kwargs):
            if text == "abCD":
                return dict(input_ids=[97, 1000, 68], offset_mapping=[(0, 1), (1, 3), (3, 4)])
            return super().__call__(text)
    tokenizer = CrossTokenizer()
    layout = SimpleNamespace(ids=CharTokenizer().encode("XYabCDZ"), initial=2, chunks=[[2, 3], [4, 5]])
    ownership = inspect_plan(tokenizer, layout, [0, 1], sink_tokens=2)["canonical_ownership"]
    assert ownership["stable_chunk_boundaries"] and ownership["ambiguous_chunk_tokens"] == 1
    assert ownership["uncovered_nonspace_characters"] == 2
    plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, [0, 1], [10, 10])
    mapped = selected_chunk_positions(layout, [0, 1], plan, 2)
    with pytest.raises(ValueError, match="multiple selected chunks"):
        build_canonical_evidence(tokenizer, layout, [0, 1], plan, mapped, sink_tokens=2)


def test_utf8_split_boundary_is_reported_unsupported():
    class ByteTokenizer:
        def encode(self, text, **_kwargs):
            return list(text.encode("utf-8"))
        def decode(self, ids, **_kwargs):
            return bytes(ids).decode("utf-8", errors="replace")
    tokenizer = ByteTokenizer()
    layout = SimpleNamespace(ids=list("XYaéZ".encode("utf-8")), initial=2, chunks=[[2, 3], [4]])
    ownership = inspect_plan(tokenizer, layout, [0, 1], sink_tokens=2)["canonical_ownership"]
    assert not ownership["stable_chunk_boundaries"] and ownership["ambiguous_chunk_tokens"] is None
    plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, [0, 1], [10, 10])
    mapped = selected_chunk_positions(layout, [0, 1], plan, 2)
    with pytest.raises(ValueError, match="splits a decoded character"):
        build_canonical_evidence(tokenizer, layout, [0, 1], plan, mapped, sink_tokens=2)


def test_canonical_evidence_refuses_to_modify_sink_positions():
    tokenizer = CharTokenizer()
    layout = SimpleNamespace(ids=tokenizer.encode("XYabcdefghZ"), initial=2,
                             chunks=[[2, 3, 4, 5], [6, 7], [8, 9]])
    plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, [1], [10, 10])
    mapped = selected_chunk_positions(layout, [1], plan, 2)
    with pytest.raises(ValueError, match="attention sink"):
        build_canonical_evidence(tokenizer, layout, [1], plan, mapped, sink_tokens=5)


def test_empty_summary_fails_closed():
    with pytest.raises(ValueError, match="At least one"):
        summarize([])


def test_lightweight_mapping_matches_production_across_discontinuous_chunks():
    rng = random.Random(41)
    for _ in range(100):
        cursor, chunks = 3, []
        for _ in range(rng.randint(1, 8)):
            cursor += rng.randint(0, 2)
            size = rng.randint(1, 8)
            chunks.append(list(range(cursor, cursor + size)))
            cursor += size
        layout = SimpleNamespace(ids=list(range(cursor + 3)), initial=3, chunks=chunks)
        selected = sorted(rng.sample(range(len(chunks)), rng.randint(1, len(chunks))))
        plan = selected_read_plan(layout.ids, chunks, layout.initial, selected, [1000, 1001])
        assert selected_chunk_positions(layout, selected, plan, 2) == production_mapping(layout, selected, plan, 2)


def test_lightweight_gpt_template_matches_production_fixed_date(monkeypatch):
    class Template(CharTokenizer):
        def apply_chat_template(self, _messages, **_kwargs):
            return "<|start|>system<|message|>Current date: 2026-10-20\n<|end|><|start|>assistant"
    tokenizer = Template()
    monkeypatch.setenv("MASKED_TTT_FIXED_CHAT_DATE", "2026-11-03")
    production = ProductionFinalChannelTokenizer(tokenizer)
    lightweight = FinalChannelTokenizer(tokenizer, "2026-11-03")
    with production.request_scope():
        for tokenize in (False, True):
            kwargs = dict(tokenize=tokenize, add_generation_prompt=True, enable_thinking=False)
            assert lightweight.apply_chat_template([], **kwargs) == production.apply_chat_template([], **kwargs)
