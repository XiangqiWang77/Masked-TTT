from types import SimpleNamespace
import random

import pytest

from experiments.value_lr_physical.audit_selected_tokenization import (FinalChannelTokenizer,
    inspect_plan, selected_chunk_positions, summarize)
from masked_ttt._layout import selected_read_plan
from masked_ttt.architectures import FinalChannelTokenizer as ProductionFinalChannelTokenizer
from masked_ttt.standalone import selected_chunk_positions as production_mapping


class CharTokenizer:
    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **_kwargs):
        return "".join(chr(value) for value in ids)


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
    tokenizer = MergeTokenizer()
    layout = SimpleNamespace(ids=[88, 89, 97, 98, 99, 100, 90], initial=2,
                             chunks=[[2, 3], [4, 5]])
    row = inspect_plan(tokenizer, layout, [0], sink_tokens=2)
    assert not row["canonical_identical"] and row["canonical_length_delta"] == -1
    assert row["exact_selected_token_mapping"] and row["replacement_characters"] == 0


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
