import math

import pytest

from .selector import choose


def input_for(chunks,scores):
    document='\n'.join(chunks);spans=[];at=0
    for chunk in chunks:
        spans.append((at,at+len(chunk)));at+=len(chunk)+1
    records=[dict(chunk=i,utility_output_valid=True,decision_probabilities={'yes':score})
             for i,score in enumerate(scores)]
    return document,spans,records


def test_entity_link_recovers_related_event_without_using_labels():
    chunks=['Unrelated text.']*8
    chunks[1]='Daniel grabbed the milk.'
    chunks[2]='Daniel moved to the garden.'
    chunks[6]='Daniel dropped the milk.'
    scores=[.01,.9,.05,.8,.7,.6,.85,.01]
    document,spans,records=input_for(chunks,scores)
    baseline=choose(document,'Where is the milk?',spans,records,.5,'score')
    linked=choose(document,'Where is the milk?',spans,records,.5,'entity')
    assert 2 not in baseline and 2 in linked
    assert 1 in linked and 6 in linked and len(linked)==4


def test_every_policy_uses_exactly_the_same_budget_and_stable_order():
    document,spans,records=input_for(['Mary went home.']*17,[.5]*17)
    for policy in ('score','neighbor','entity'):
        first=choose(document,'Where is Mary?',spans,records,.2,policy)
        assert first==choose(document,'Where is Mary?',spans,records,.2,policy)
        assert first==sorted(set(first)) and len(first)==math.ceil(17*.2)


def test_invalid_data_is_rejected_and_small_budget_keeps_highest_scores():
    document,spans,records=input_for(['A short text.']*3,[.1,.2,.9])
    assert choose(document,'What?',spans,records,.1,'entity')==[2]
    records[2]['utility_output_valid']=False
    with pytest.raises(ValueError):choose(document,'What?',spans,records,.5,'score')
