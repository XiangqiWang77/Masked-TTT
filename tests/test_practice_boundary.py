import json
from types import SimpleNamespace

import torch

from masked_ttt.config import TTTConfig
from masked_ttt.request import Request
import masked_ttt.self_study_runtime as study


def test_verified_practice_uses_original_separator_only_on_boundary_mismatch(monkeypatch):
    document='Other notes. X final budget is 320.'
    request=Request(document=document,question='How much did X finally receive?',
        prompt='Context:\n'+document+'\nQuestion: How much did X finally receive?',max_new_tokens=4)
    layout=SimpleNamespace(spans=[(0,len(document))])
    proposal=json.dumps({'exercises':[dict(question='What is the final X budget?',
        answer='320',evidence='X final budget is 320.')]})
    seen=[]
    def suffixes(_engine,_layout,_selected,_teacher,new_request):
        seen.append(new_request.prompt)
        if '\n\nPractice question:' in new_request.prompt:
            raise ValueError('Practice prompt changed the original document token prefix')
        return torch.tensor([1]),SimpleNamespace(document_cache=None,document_end=1,
                                                 question_tokens=torch.tensor([1]))
    monkeypatch.setattr(study,'suffixes',suffixes)
    monkeypatch.setattr(study,'generate_from_cache',lambda *a,**k:([1],proposal))
    monkeypatch.setattr(study,'answer_quote',lambda *a,**k:'320')
    config=TTTConfig(self_question_count=1,practice_boundary_fallback=True)
    engine=SimpleNamespace(config=config,synchronize=lambda:None)
    paths,info=study.generate_self_study(engine,None,None,request,layout,[0])
    assert len(paths)==1 and info['verified']==1 and info['boundary_fallback_recovered']==1
    assert info['checks'][0]['boundary_fallback_used']
    assert not info['reference_answers_used'] and not info['student_answer_hint_used']
    assert any(document+'\nPractice question:' in text for text in seen)
    assert all('320' not in text.split('Practice question:')[-1] for text in seen
               if 'Practice question:' in text)


def test_boundary_fallback_never_ignores_a_different_prefix_error(monkeypatch):
    document='X final budget is 320.'
    request=Request(document=document,question='Original?',prompt='Context: '+document+'\nQuestion: Original?')
    proposal=json.dumps({'exercises':[dict(question='Final X?',answer='320',evidence=document)]})
    def suffixes(_engine,_layout,_selected,_teacher,new_request):
        if 'Practice question:' in new_request.prompt:
            raise ValueError('Practice prompt changed the selected Teacher document prefix')
        return torch.tensor([1]),SimpleNamespace(document_cache=None,document_end=1,
                                                 question_tokens=torch.tensor([1]))
    monkeypatch.setattr(study,'suffixes',suffixes)
    monkeypatch.setattr(study,'generate_from_cache',lambda *a,**k:([1],proposal))
    monkeypatch.setattr(study,'answer_quote',lambda *a,**k:'320')
    engine=SimpleNamespace(config=TTTConfig(self_question_count=1,practice_boundary_fallback=True),
                           synchronize=lambda:None)
    paths,info=study.generate_self_study(engine,None,None,request,
        SimpleNamespace(spans=[(0,len(document))]),[0])
    assert paths==[] and info['verified']==0 and info['boundary_fallback_recovered']==0
    assert info['checks'][0]['reason']=='document_prefix_mismatch'
