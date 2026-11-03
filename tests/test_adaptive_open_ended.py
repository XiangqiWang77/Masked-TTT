"""Extra QA steps are bounded, optional and use the same on-policy target."""
from dataclasses import replace

import pytest

from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
import masked_ttt.pipeline as pipeline


@pytest.mark.parametrize('enabled,threshold,expected',[
    (False,0.,2),
    (True,0.,5),
    (True,100.,2),
])
def test_open_ended_stopping_uses_current_teacher_kl(tiny_backbone,monkeypatch,enabled,threshold,expected):
    tiny_backbone.config=replace(tiny_backbone.config,steps=2,max_answer_steps=5,
        self_question_count=0,guard_enabled=False,standalone_cartridge_enabled=False,
        adaptive_open_ended=enabled,adaptive_open_ended_kl_stop=threshold)
    class Selector:
        def prepare(self,backbone,request):
            return PreparedRequest(request,prepare_layout(backbone,request),[1],dict(backend='test'))
    # Keep a differentiable positive objective so the zero-threshold test runs
    # the complete budget on arbitrary tiny random model parameters.
    original=pipeline.distribution_kl
    monkeypatch.setattr(pipeline,'distribution_kl',lambda logits,target:original(logits,target)+1.)
    request=Request(document='The red folder is old. The blue folder is new. '*2,
                    question='Which folder is new?',max_new_tokens=3)
    result=MaskedTTT(tiny_backbone,Selector()).answer(request)
    assert result['training']['accepted_steps']==expected
    assert all(step['question_path']=='main' for step in result['training']['steps'])
    assert result['training']['adaptive_open_ended'] is enabled
    assert result['training']['teacher_conditioned_on_student_prefix']
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['final_question_recomputed'] and not result['teacher_answer_in_final_prompt']
