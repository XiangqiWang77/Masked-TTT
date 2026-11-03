from dataclasses import replace
import json
import weakref

import pytest

from masked_ttt import MaskedTTT, Request, TTTConfig
import masked_ttt.pipeline as pipeline
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
import masked_ttt.self_study_runtime as study
from test_direct_cache import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize('factory',[make_engine,gpt_engine])
@pytest.mark.parametrize('practice',[False,True])
@pytest.mark.parametrize('use_judge',[False,True])
def test_releasing_full_cache_preserves_training_and_outputs(factory,practice,use_judge,monkeypatch):
    engine=factory()
    engine.config=replace(engine.config,chunk_tokens=64,self_question_count=int(practice),
                          matched_cache_direct=True,practice_boundary_fallback=True,
                          guard_enabled=use_judge,teacher_judge_policy='grounded_veto' if use_judge else 'off',
                          teacher_veto_against_native=use_judge)
    evidence='X final budget is 320.'
    document='Other notes. '+evidence+' More filler. '*8
    request=Request(document=document,question='How much did X finally receive?',max_new_tokens=3)
    layout=prepare_layout(engine,request)
    selected=[next(i for i,(a,b) in enumerate(layout.spans) if evidence in document[a:b])]
    class Selector:
        def prepare(self,backbone,original):
            return PreparedRequest(original,prepare_layout(backbone,original),selected,dict(backend='synthetic'))
    if practice:
        proposal=json.dumps({'exercises':[dict(question='What is the final X budget?',answer='320',
                                               evidence=evidence)]})
        monkeypatch.setattr(study,'generate_from_cache',lambda *args,**kwargs:([1],proposal))
        monkeypatch.setattr(study,'answer_quote',lambda *args,**kwargs:'320')
    seen=[];prefill=pipeline.prefill_document;train=pipeline.train_selected_cache
    def capture(*args,**kwargs):
        result=prefill(*args,**kwargs);seen.append(weakref.ref(result));return result
    def check_release(engine,*args,**kwargs):
        assert (seen[-1]() is None)==engine.config.release_full_cache_before_direct
        return train(engine,*args,**kwargs)
    monkeypatch.setattr(pipeline,'prefill_document',capture)
    monkeypatch.setattr(pipeline,'train_selected_cache',check_release)
    judged=[]
    if use_judge:
        def judge(engine,reader,cache,*args,**kwargs):
            assert seen[-1]() is cache;judged.append('teacher')
            return dict(apply_update=True,reference_answers_used=False,seconds=0.)
        def compare(engine,reader,cache,*args,**kwargs):
            assert seen[-1]() is cache;judged.append('native')
            return dict(use_adapted=True,reference_answers_used=False,seconds=0.)
        monkeypatch.setattr(pipeline,'judge_teacher',judge)
        monkeypatch.setattr(pipeline,'guard_answers',compare)
    old=MaskedTTT(engine,Selector()).answer(request)
    engine.config=replace(engine.config,release_full_cache_before_direct=True)
    new=MaskedTTT(engine,Selector()).answer(request)
    assert not old['memory']['full_cache_released_before_update']
    assert new['memory']['full_cache_released_before_update']
    assert new['teacher']['tokens']==old['teacher']['tokens']
    assert new['self_study']['verified']==old['self_study']['verified']==int(practice)
    for name in ('zero_update','trained','zero_format','trained_format'):
        assert new['matched_cache_study'][name]['generated_token_ids']==old['matched_cache_study'][name]['generated_token_ids']
    for before,after in zip(old['training']['steps'],new['training']['steps']):
        assert before['student_rollout_tokens']==after['student_rollout_tokens']
        assert before['target_positions']==after['target_positions']
        for name in ('teacher_KL_before','gain_gradient_norm','value_gradient_norm'):
            assert before[name]==pytest.approx(after[name],rel=1e-5,abs=1e-7)
    assert old['backbone_unchanged'] and new['backbone_unchanged']
    assert judged==(['teacher','native']*2 if use_judge else [])
    assert old['original_document_KV_unchanged'] and new['original_document_KV_unchanged']
    assert new['matched_cache_study']['source_selected_cache_unchanged']


def test_release_requires_direct_mode_and_boolean_value():
    with pytest.raises(ValueError,match='requires direct'):
        TTTConfig(release_full_cache_before_direct=True)
    with pytest.raises(ValueError,match='must be boolean'):
        TTTConfig(release_full_cache_before_direct='yes')


@pytest.mark.parametrize('factory',[make_engine,gpt_engine])
def test_real_teacher_judgment_does_not_retain_full_cache(factory,monkeypatch):
    engine=factory()
    engine.config=replace(engine.config,matched_cache_direct=True,release_full_cache_before_direct=True,
        guard_enabled=True,teacher_judge_policy='grounded_veto',teacher_veto_against_native=True,
        teacher_judge_focused_tokens=4,teacher_judge_full_tokens=4)
    request=Request(document='The old report was red. The final report was blue. '*3,
                    question='What color was the final report?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,original):
            layout=prepare_layout(backbone,original)
            return PreparedRequest(original,layout,[len(layout.chunks)//2],dict(backend='synthetic'))
    seen=[];prefill=pipeline.prefill_document;train=pipeline.train_selected_cache
    def capture(*args,**kwargs):
        result=prefill(*args,**kwargs);seen.append(weakref.ref(result));return result
    def check_release(engine,*args,**kwargs):
        assert seen[-1]() is None
        return train(engine,*args,**kwargs)
    monkeypatch.setattr(pipeline,'prefill_document',capture)
    monkeypatch.setattr(pipeline,'train_selected_cache',check_release)
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result['teacher_judgment']['executed_before_optimizer']
    assert result['memory']['full_cache_released_before_update']
    assert result['original_document_KV_unchanged'] and result['backbone_unchanged']
