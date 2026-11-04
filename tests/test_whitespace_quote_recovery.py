import json
from types import SimpleNamespace

import pytest

from experiments.whitespace_quote_recovery.parser import (
    original_parse, original_source_quote, parse_with_quote_recovery, use_candidate)


DOCUMENT = "Mary moved   to the kitchen.\nJohn went to the garden."
FIRST = "Mary moved   to the kitchen."
SECOND = "John went to the garden."
SPANS = [(0, len(FIRST)), (len(FIRST) + 1, len(DOCUMENT))]
VARIANT = {"question": "Where did Mary go?", "answer": "kitchen", "evidence": "Mary moved to the kitchen."}
STRICT = {"question": "Where did John go?", "answer": "garden", "evidence": SECOND}


def parse(items, selected=(0, 1), limit=2):
    return parse_with_quote_recovery(json.dumps({"exercises": items}), DOCUMENT, SPANS,
                                     selected, "What happened next?", limit)


def test_recovers_only_whitespace_and_maps_back_to_exact_selected_source():
    result, rejections = parse([VARIANT], selected=(0,))
    assert len(result) == 1 and result[0].evidence == FIRST and result[0].source_chunk == 0
    assert not rejections
    assert parse([VARIANT], selected=(1,))[0] == []


def test_retains_strict_items_first_and_obeys_limit_and_deduplication():
    strict, _ = original_parse(json.dumps({"exercises": [VARIANT, STRICT]}), DOCUMENT, SPANS,
                               (0, 1), "What happened next?", 2)
    result, _ = parse([VARIANT, STRICT, VARIANT])
    assert result[0] == strict[0] and [x.answer for x in result] == ["garden", "kitchen"]
    assert parse([VARIANT, STRICT], limit=1)[0] == strict


def test_does_not_change_words_punctuation_or_cross_chunk_boundaries():
    changed = dict(VARIANT, evidence="Mary moved to the office.")
    assert parse([changed])[0] == []
    assert original_source_quote("Mary moved to the kitchen. John went", DOCUMENT, SPANS, (0, 1)) is None
    assert original_source_quote("mary moved to the kitchen.", DOCUMENT, SPANS, (0,)) is None


def test_numeric_boundaries_and_answer_containment_are_kept():
    decimal = "Budget   is 320.5"
    assert original_source_quote("Budget is 320.", decimal, [(0, len(decimal))], (0,)) is None
    negative = "−320   budget"
    assert original_source_quote("320 budget", negative, [(0, len(negative))], (0,)) is None
    bad = dict(VARIANT, answer="garden")
    assert parse([bad])[0] == []


def test_scoped_runtime_restores_parser_on_failure():
    runtime = SimpleNamespace(parse_exercises=original_parse)
    with pytest.raises(RuntimeError):
        with use_candidate(runtime):
            assert runtime.parse_exercises is parse_with_quote_recovery
            raise RuntimeError("test")
    assert runtime.parse_exercises is original_parse


def test_recovered_quote_still_needs_independent_model_verification(monkeypatch):
    import masked_ttt.self_study_runtime as runtime
    from masked_ttt.request import Request

    text = json.dumps({"exercises": [VARIANT]})
    focused = SimpleNamespace(document_cache=None, document_end=1, question_tokens=[1])
    monkeypatch.setattr(runtime, "suffixes", lambda *args, **kwargs: ("student", focused))
    monkeypatch.setattr(runtime, "generate_from_cache", lambda *args, **kwargs: ([1], text))
    verifier = SimpleNamespace(answer="garden")
    monkeypatch.setattr(runtime, "answer_quote", lambda *args, **kwargs: verifier.answer)
    engine = SimpleNamespace(config=SimpleNamespace(self_question_count=2, self_question_max_tokens=192,
        self_question_verify_tokens=16, student_rollout_tokens=8), synchronize=lambda: None)
    request = Request(document=DOCUMENT, question="What happened next?", prompt=DOCUMENT+"\nWhat happened next?")
    with use_candidate(runtime):
        paths, info = runtime.generate_self_study(engine, None, None, request, SimpleNamespace(spans=SPANS), (0,))
        assert paths == [] and info["parsed"] == 1 and info["verified"] == 0
        verifier.answer = "kitchen"
        paths, info = runtime.generate_self_study(engine, None, None, request, SimpleNamespace(spans=SPANS), (0,))
        assert len(paths) == 1 and info["verified"] == 1
