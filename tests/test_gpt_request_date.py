from types import SimpleNamespace
from contextvars import Context

import pytest

import masked_ttt.architectures as architectures
from masked_ttt.architectures import FinalChannelTokenizer
from masked_ttt.pipeline import MaskedTTT


class ClockedTokenizer:
    eos_token_id = 2

    def __init__(self, today):
        self.today = today

    def apply_chat_template(self, messages, **kwargs):
        return ("<|start|>system<|message|>Knowledge cutoff: 2024-06\n"
                f"Current date: {self.today}\n\nReasoning: low<|end|>"
                "<|start|>user<|message|>" + messages[0]["content"] +
                "<|end|><|start|>assistant")

    def encode(self, text, **kwargs):
        return list(text.encode())


def render(tokenizer, document):
    return tokenizer.apply_chat_template([dict(role="user", content=document)],
                                         tokenize=False, add_generation_prompt=True)


def test_request_scope_pins_system_date_across_midnight_and_preserves_user_text(monkeypatch):
    monkeypatch.setattr(architectures, "_template_date", lambda: "2026-11-03")
    underlying = ClockedTokenizer("2026-11-03")
    tokenizer = FinalChannelTokenizer(underlying)
    document = "Source says:\nCurrent date: 1999-01-01\nThe final answer is blue."
    with tokenizer.request_scope():
        first = render(tokenizer, document)
        underlying.today = "2026-11-04"
        second = render(tokenizer, document)
        assert first == second
        assert "Current date: 2026-11-03\n" in second
        assert "Current date: 1999-01-01\n" in second
        assert tokenizer.masked_ttt_request_date == "2026-11-03"
    assert tokenizer.masked_ttt_request_date is None
    assert "Current date: 2026-11-04\n" in render(tokenizer, document)


def test_answer_scope_resets_after_success_and_failure(monkeypatch):
    monkeypatch.setattr(architectures, "_template_date", lambda: "2026-11-03")
    tokenizer = FinalChannelTokenizer(ClockedTokenizer("2026-11-03"))
    method = MaskedTTT(SimpleNamespace(tokenizer=tokenizer), selector=object())

    def inside(_request, **_kwargs):
        assert tokenizer.masked_ttt_request_date == "2026-11-03"
        tokenizer.tokenizer.today = "2026-11-04"
        assert "Current date: 2026-11-03\n" in render(tokenizer, "text")
        return "ok"

    monkeypatch.setattr(method, "_answer", inside)
    assert method.answer(object()) == "ok" and tokenizer.masked_ttt_request_date is None
    def fail(_request, **_kwargs):
        assert tokenizer.masked_ttt_request_date == "2026-11-03"
        raise RuntimeError("failure")
    monkeypatch.setattr(method, "_answer", fail)
    with pytest.raises(RuntimeError): method.answer(object())
    assert tokenizer.masked_ttt_request_date is None


def test_fixed_benchmark_date_is_validated(monkeypatch):
    monkeypatch.setenv("MASKED_TTT_FIXED_CHAT_DATE", "2026-11-03")
    assert architectures._template_date() == "2026-11-03"
    monkeypatch.setenv("MASKED_TTT_FIXED_CHAT_DATE", "not-a-date")
    with pytest.raises(ValueError): architectures._template_date()


def test_independent_contexts_do_not_share_request_dates(monkeypatch):
    clock = {"date": "2026-11-03"}
    monkeypatch.setattr(architectures, "_template_date", lambda: clock["date"])
    tokenizer = FinalChannelTokenizer(ClockedTokenizer("2026-11-03"))
    def other_request():
        with tokenizer.request_scope():
            assert tokenizer.masked_ttt_request_date == "2026-11-04"
    with tokenizer.request_scope():
        clock["date"] = "2026-11-04"
        Context().run(other_request)
        assert tokenizer.masked_ttt_request_date == "2026-11-03"
    assert tokenizer.masked_ttt_request_date is None
