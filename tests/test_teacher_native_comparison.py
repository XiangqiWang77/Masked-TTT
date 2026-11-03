"""Select Teacher supervision before training using the original full context."""
from dataclasses import replace

import pytest

from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
import masked_ttt.pipeline as pipeline


@pytest.mark.parametrize('veto,expected_steps',[(True,0),(False,2)])
def test_native_comparison_precedes_any_student_update(tiny_backbone,monkeypatch,veto,expected_steps):
    tiny_backbone.config=replace(tiny_backbone.config,self_question_count=0,guard_enabled=True,
        standalone_cartridge_enabled=False,teacher_veto_against_native=True,
        teacher_veto_minimum_confidence=.8,teacher_veto_minimum_label_mass=.2)
    class Selector:
        def prepare(self,backbone,request):
            return PreparedRequest(request,prepare_layout(backbone,request),[1],dict(backend='test'))
    original_teacher=pipeline.build_teacher
    def teacher(*args):
        result=original_teacher(*args);result.prediction='teacher candidate';result.complete=True;return result
    monkeypatch.setattr(pipeline,'build_teacher',teacher)
    monkeypatch.setattr(pipeline,'generate_from_cache',lambda *args:([33,2],'native candidate'))
    events=[]
    def guard(engine,reader,cache,layout,request,base,adapted):
        if adapted=='teacher candidate':
            events.append('compare_before_training')
            assert engine.config.guard_policy=='veto'
            assert engine.config.guard_minimum_confidence==.8
            return dict(use_adapted=not veto,seconds=0.,reference_answers_used=False,
                        reason='baseline_preferred_in_both_orders' if veto else 'keep_student')
        events.append('final_guard')
        return dict(use_adapted=False,seconds=0.,reference_answers_used=False,reason='test')
    monkeypatch.setattr(pipeline,'guard_answers',guard)
    original_rollout=pipeline.student_rollout
    def rollout(*args,**kwargs):
        assert events and events[0]=='compare_before_training'
        events.append('train');return original_rollout(*args,**kwargs)
    monkeypatch.setattr(pipeline,'student_rollout',rollout)
    request=Request(document='The red folder is old. The blue folder is new. '*2,
                    question='Which folder is new?',max_new_tokens=3)
    result=MaskedTTT(tiny_backbone,Selector()).answer(request)
    assert result['training']['accepted_steps']==expected_steps
    assert result['teacher_judgment']['native_comparison']['reference_answers_used'] is False
    assert result['teacher_judgment']['native_comparison_config']==dict(
        policy='veto',minimum_confidence=.8,minimum_label_mass=.2)
    assert result['teacher_judgment']['main_signal_accepted'] is (not veto)
    assert result['training']['teacher_main_signal_used'] is (not veto)
    assert result['training']['teacher_veto_against_native'] is True
    assert result['original_document_KV_unchanged'] and result['backbone_unchanged']
    assert events[0]=='compare_before_training' and events[-1]=='final_guard'


def test_teacher_comparison_requires_a_native_answer(tiny_backbone):
    with pytest.raises(ValueError):
        replace(tiny_backbone.config,guard_enabled=False,teacher_veto_against_native=True)
