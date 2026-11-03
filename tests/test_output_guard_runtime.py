"""Order handling and conservative fallback for the full-context judge."""
from types import SimpleNamespace

import pytest
from masked_ttt.output_guard_runtime import guard_answers,label_ids
import masked_ttt.output_guard_runtime as guard


class Tokenizer:
    def encode(self,text,**kwargs):
        return {'A':[1],' A':[11],'B':[2],' B':[12],'U':[3],' U':[13]}[text]


def test_label_token_sets_are_unique_and_support_space_variants():
    assert label_ids(Tokenizer(),20)=={'A':{1,11},'B':{2,12},'U':{3,13}}
    class Ambiguous(Tokenizer):
        def encode(self,text,**kwargs):return [1]
    with pytest.raises(ValueError):label_ids(Ambiguous(),20)


def test_judge_receives_both_orderings_and_accepts_only_consensus(monkeypatch):
    engine=SimpleNamespace(config=SimpleNamespace(guard_minimum_confidence=.65,
                                                  guard_minimum_label_mass=.2),
                           synchronize=lambda:None)
    calls=[]
    def review(_engine,_reader,_cache,_layout,_request,first,second):
        calls.append((first,second))
        return (dict(A=.1,B=.8,U=.1,label_mass=.7) if len(calls)==1
                else dict(A=.8,B=.1,U=.1,label_mass=.7))
    monkeypatch.setattr(guard,'review_order',review)
    result=guard_answers(engine,None,None,None,None,'baseline','adapted')
    assert result['use_adapted'] and calls==[('baseline','adapted'),('adapted','baseline')]
    assert result['reference_answers_used'] is False


def test_no_judge_when_answers_match_and_unavailable_judge_falls_back(monkeypatch):
    engine=SimpleNamespace(config=SimpleNamespace(guard_minimum_confidence=.65,
                                                  guard_minimum_label_mass=.2),
                           synchronize=lambda:None)
    def unavailable(*args):raise ValueError('too long')
    monkeypatch.setattr(guard,'review_order',unavailable)
    assert guard_answers(engine,None,None,None,None,'same','same')['reason']=='same_answer'
    result=guard_answers(engine,None,None,None,None,'baseline','adapted')
    assert not result['use_adapted'] and result['reason']=='judge_unavailable'
