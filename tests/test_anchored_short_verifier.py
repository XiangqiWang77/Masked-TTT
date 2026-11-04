from dataclasses import replace
import json

import pytest

from experiments.anchored_short_verifier.verifier import accepted,anchored_short_verifier
from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from masked_ttt.self_study import Exercise
from test_standalone import make_engine


def exercise(question,answer,evidence=None):
    return Exercise(question,answer,evidence if evidence is not None else answer,0)


def test_original_exact_verification_is_preserved():
    ok,reason=accepted(exercise('Where did John go?','garden','John went to the garden.'),'The garden.')
    assert ok and reason=='original_exact'


@pytest.mark.parametrize('question,proposal,verified',[
    ('Where did John go?','John journeyed to the garden.','The garden.'),
    ('Who moved to the kitchen?','Mary moved to the kitchen.','Mary.'),
    ('What is north of the bedroom?','The hallway is north of the bedroom.','The hallway.'),
])
def test_verified_unique_short_span_inside_the_anchored_proposal(question,proposal,verified):
    ok,reason=accepted(exercise(question,proposal),verified)
    assert ok and reason=='verified_anchored_subspan'


@pytest.mark.parametrize('question,proposal,verified,evidence',[
    ('Where did Mary go first?','Mary moved to the kitchen.','kitchen','Mary moved to the kitchen.'),
    ('Where did John drop the apple?','John dropped the apple there.','there','John dropped the apple there.'),
    ('Where is Mary?','Mary was not in the kitchen.','kitchen','Mary was not in the kitchen.'),
    ('Where is Mary not located?','Mary is in the kitchen.','kitchen','Mary is in the kitchen.'),
    ('What was the budget?','X budget was 320, Y budget was 250.','320','X budget was 320, Y budget was 250.'),
    ('Where is John?','John went to the garden.','unknown','John went to the garden.'),
    ('Where is John?','John went to the garden.','garden','John went to the office.'),
    ('Where are they?','John and Mary visited the garden and the garden.','garden',None),
])
def test_ambiguous_negated_unanchored_and_temporal_proposals_are_not_newly_accepted(
        question,proposal,verified,evidence):
    assert not accepted(exercise(question,proposal,evidence),verified)[0]


def test_new_verifier_adds_only_selected_quote_based_practice(monkeypatch):
    import masked_ttt.self_study_runtime as runtime
    document='John journeyed to the garden. '+'Other material. '*12
    request=Request(document=document,question='Where is John now?',max_new_tokens=3)
    engine=make_engine()
    engine.config=replace(engine.config,chunk_tokens=128,self_question_count=1,
        guard_enabled=False,selected_transfer_weight=0.,release_full_before_cartridge=True)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,list(range(len(layout.chunks))),dict(backend='unit_test'))
    quote='John journeyed to the garden.'
    payload=json.dumps({'exercises':[dict(question='Where did John journey to?',answer=quote,evidence=quote)]})
    monkeypatch.setattr(runtime,'generate_from_cache',lambda *args,**kwargs:([1],payload))
    monkeypatch.setattr(runtime,'answer_quote',lambda *args,**kwargs:'The garden.')
    method=MaskedTTT(engine,Selector());original=method.answer(request)
    with anchored_short_verifier() as stats:
        result=method.answer(request)
    assert original['self_study']['verified']==0 and result['self_study']['verified']==1
    assert stats['verified_anchored_subspan']==1 and stats['rejected']==0
    assert [row['question_path'] for row in result['training']['steps']]==['main','practice_1','main']
    assert result['teacher']['tokens']==original['teacher']['tokens']
    assert result['training']['objective']=='on-policy KL(teacher || student)'
    assert result['training']['selected_transfer_weight']==0
    assert not result['self_study']['reference_answers_used']
    assert not result['self_study']['student_answer_hint_used']
    assert not result['teacher_answer_in_final_prompt']
    assert result['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']==0
    assert result['original_document_KV_unchanged'] and result['backbone_unchanged']


def test_runtime_hooks_are_restored_on_failure():
    import masked_ttt.self_study_runtime as runtime
    quote=runtime.answer_quote;compare=runtime.short_answer_equivalent
    with pytest.raises(RuntimeError,match='intentional'):
        with anchored_short_verifier():raise RuntimeError('intentional')
    assert runtime.answer_quote is quote and runtime.short_answer_equivalent is compare
