import pytest

from experiments.babilong_linked_selection import focused_question_selector as subject
from experiments.babilong_linked_selection import focused_question_coverage as audit


def test_routes_from_question_text_only(monkeypatch):
    monkeypatch.setattr(subject, "choose", lambda *_a, **_kw: [1, 3])
    monkeypatch.setattr(subject, "question_overlap", lambda *_a, **_kw: [1, 4])
    def select(question, **kwargs):
        return subject.focused_question_overlap("doc", question, [], [], .1, **kwargs)
    assert select("Where is Mary?") == [1, 4]
    assert select("What is Mary carrying?") == [1, 4]
    assert select("What did Mary give to John?") == [1, 3]
    assert select("What did Mary give to John?", include_give=True) == [1, 4]
    assert select("Who gave the apple to John?", include_give=True) == [1, 4]
    assert select("Is Mary in the kitchen?", include_give=True) == [1, 3]
    with pytest.raises(ValueError):
        select("Where is Mary?", include_give=1)


def test_audit_restores_selector_and_reports_increment_against_where(monkeypatch, tmp_path):
    original = audit.base.question_overlap
    fields = ("complete", "changed_selections", "newly_complete", "lost_complete",
              "full_only_newly_complete", "teacher_only_lost_complete")
    def stats(value):
        return {field: value for field in fields}
    def fake(_workspace):
        assert audit.base.question_overlap is not original
        return dict(cases=200, source_questions=50, fraction=.1,
                    groups={"all": dict(entity25=stats(0), proper_where=stats(1),
                                        mix=stats(2), lexical=stats(4))})
    monkeypatch.setattr(audit.base, "collect", fake)
    monkeypatch.setattr(audit.base, "decision", lambda _: {"eligible_for_model_test": True})
    result = audit.collect(tmp_path)
    assert audit.base.question_overlap is original
    assert result["increment_vs_where"]["where_carry"]["complete"] == 1
    assert result["increment_vs_where"]["where_carry_give"]["complete"] == 3
    assert result["selection_accepts_no_labels"] and result["online_api_calls"] == 0
    def fail(_workspace):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(audit.base, "collect", fail)
    with pytest.raises(RuntimeError):
        audit.collect(tmp_path)
    assert audit.base.question_overlap is original
