import json
from types import SimpleNamespace

import pytest

from experiments.truncated_study_json.parser import (
    complete_items, original_parse, parse_with_truncated_recovery, use_candidate)


DOCUMENT = "Mary went to the kitchen. John moved to the office."
SPANS = [(0, 25), (25, len(DOCUMENT))]
ITEM = {"question": "Where did Mary go?", "answer": "kitchen", "evidence": "Mary went to the kitchen."}


def run(text, selected=(0,), question="Where is John?"):
    return parse_with_truncated_recovery(text, DOCUMENT, SPANS, selected, question, 2)


def test_recovers_only_complete_item_from_truncated_array():
    text = '{"exercises":[' + json.dumps(ITEM) + ', {"question":"Where did John'
    result, rejected = run(text)
    assert [x.answer for x in result] == ["kitchen"]
    assert rejected == {}
    assert len(complete_items(text)) == 1


def test_wont_repair_incomplete_item_or_read_arbitrary_objects():
    assert run('{"exercises":[{"question":"Where did Mary go?"')[0] == []
    assert run(json.dumps(ITEM))[0] == []
    assert complete_items('{"other":[' + json.dumps(ITEM) + ']') == []


def test_original_filters_apply_to_recovered_items():
    text = '{"exercises":[' + json.dumps(ITEM)
    assert run(text, selected=(1,))[1] == {"quote_not_in_selected_chunk": 1}
    assert run(text, question=ITEM["question"])[1] == {"duplicate_or_original_question": 1}
    bad = dict(ITEM, answer="office")
    assert run('{"exercises":[' + json.dumps(bad))[1] == {"answer_not_in_quote": 1}


def test_valid_json_behavior_is_identical_to_original():
    text = json.dumps({"exercises": [ITEM, ITEM]})
    expected = original_parse(text, DOCUMENT, SPANS, (0,), "Where is John?", 2)
    assert run(text) == expected


def test_runtime_patch_restores_on_error_and_rejects_nested_patch():
    runtime = SimpleNamespace(parse_exercises=original_parse)
    with pytest.raises(RuntimeError):
        with use_candidate(runtime):
            assert runtime.parse_exercises is parse_with_truncated_recovery
            with pytest.raises(ValueError):
                with use_candidate(runtime):
                    pass
            raise RuntimeError("test")
    assert runtime.parse_exercises is original_parse


def test_recovered_practice_still_requires_independent_verification(monkeypatch):
    import masked_ttt.self_study_runtime as runtime
    from masked_ttt.request import Request

    text = '{"exercises":[' + json.dumps(ITEM)
    monkeypatch.setattr(runtime, "generate_from_cache", lambda *args, **kwargs: ([1], text))
    focused = SimpleNamespace(document_cache=None, document_end=1, question_tokens=[1])
    monkeypatch.setattr(runtime, "suffixes", lambda *args, **kwargs: ("student", focused))
    verifier = SimpleNamespace(answer="office")
    monkeypatch.setattr(runtime, "answer_quote", lambda *args, **kwargs: verifier.answer)
    engine = SimpleNamespace(config=SimpleNamespace(self_question_count=2, self_question_max_tokens=192,
        self_question_verify_tokens=16, student_rollout_tokens=8), synchronize=lambda: None)
    request = Request(document=DOCUMENT, question="Where is John?", prompt=DOCUMENT+"\nQuestion: Where is John?")
    with use_candidate(runtime):
        paths, info = runtime.generate_self_study(engine, None, None, request, SimpleNamespace(spans=SPANS), (0,))
        assert paths == [] and info["parsed"] == 1 and info["verified"] == 0
        verifier.answer = "kitchen"
        paths, info = runtime.generate_self_study(engine, None, None, request, SimpleNamespace(spans=SPANS), (0,))
        assert len(paths) == 1 and info["verified"] == 1
        assert info["reference_answers_used"] is False and info["student_answer_hint_used"] is False
