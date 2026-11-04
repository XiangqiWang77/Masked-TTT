from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments.shared_question_sd.alignment import shared_config,shared_request,verify_shared_training_suffix
from experiments.shared_question_sd.controls import aligned_controls
from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_shared_condition_keeps_teacher_and_selected_kv_but_aligns_sd(architecture):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=0.,release_full_before_cartridge=True)
    original=engine.config;instruction=original.teacher_response_instruction
    request=Request(document='Old note says red. Current note says blue. '*6,
                    question='Which note is current?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    method=MaskedTTT(engine,Selector());baseline=method.answer(request)
    aligned=shared_request(request,instruction)
    engine.config=shared_config(original)
    with verify_shared_training_suffix() as observed:
        result=method.answer(aligned)
    assert len(observed)==len(result['training']['steps'])==original.steps
    assert result['teacher']['tokens']==baseline['teacher']['tokens']
    assert result['teacher']['fresh_selected_context'] and not result['teacher']['response_instruction_applied']
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['training']['objective']=='on-policy KL(teacher || student)'
    assert result['training']['selected_transfer_weight']==0
    assert result['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']==0
    assert result['standalone_cartridge']['generated_token_ids'] and not result['teacher_answer_in_final_prompt']
    assert request.document==aligned.document and request.question==aligned.question
    assert aligned.prompt.count(instruction)==1


def test_shared_instruction_is_not_allowed_to_be_empty():
    request=Request(document='Document.',question='Question?')
    with pytest.raises(ValueError):shared_request(request,'')


def test_verified_practice_questions_keep_teacher_student_suffixes_identical(monkeypatch):
    import masked_ttt.pipeline as pipeline
    from masked_ttt.self_study import practice_request
    from masked_ttt.self_study_runtime import suffixes
    engine=make_engine()
    engine.config=replace(engine.config,self_question_count=1,guard_enabled=False,
        selected_transfer_weight=0.,release_full_before_cartridge=True)
    instruction=engine.config.teacher_response_instruction
    request=shared_request(Request(document='Old note says red. Current note says blue. '*6,
        question='Which note is current?',max_new_tokens=3),instruction)
    engine.config=shared_config(engine.config)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    def practice(backbone,reader,teacher,item,layout,selected):
        q=practice_request(item,'What color was old?',backbone.config.student_rollout_tokens)
        student,focused=suffixes(backbone,layout,selected,teacher,q)
        return [dict(name='practice_1',student_suffix=student,teacher=focused,question=q.question)],dict(
            verified=1,checks=[],reference_answers_used=False,student_answer_hint_used=False)
    monkeypatch.setattr(pipeline,'generate_self_study',practice)
    with verify_shared_training_suffix() as observed:
        result=MaskedTTT(engine,Selector()).answer(request)
    assert len(observed)==3 and [x['question_path'] for x in result['training']['steps']]==[
        'main','practice_1','main']


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_matched_control_does_not_retain_training_or_selected_source_at_final_decode(architecture):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=0.,release_full_before_cartridge=True)
    instruction=engine.config.teacher_response_instruction
    request=shared_request(Request(document='Old note is red. Current note is blue. '*6,
        question='Which color is current?',max_new_tokens=3),instruction)
    engine.config=shared_config(engine.config)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    with aligned_controls(),verify_shared_training_suffix() as observed:
        result=MaskedTTT(engine,Selector()).answer(request)
    standalone=result['standalone_cartridge'];p=standalone['aligned_controls']
    assert p['adapted']['tokens']==standalone['generated_token_ids']
    assert p['shared_question_suffix'] and p['same_selected_kv'] and len(observed)==engine.config.steps
    assert not p['teacher_answer_in_student_prompt'] and not p['reference_answers_used']
    if 'additional_source_kv_bytes_before_decode' in standalone['memory']:
        assert standalone['memory']['additional_source_kv_bytes_before_decode']==0
        assert standalone['memory']['additional_training_adapter_bytes_before_decode']==0


def test_suffix_verifier_restores_hooks_after_failure():
    import masked_ttt.pipeline as pipeline
    old_rollout=pipeline.student_rollout;old_score=pipeline.score_student_prefix
    with pytest.raises(AssertionError,match='suffixes do not match'):
        with verify_shared_training_suffix():
            pipeline.score_student_prefix(None,None,SimpleNamespace(question_tokens=torch.tensor([1])),
                                          torch.tensor([2]))
    assert pipeline.student_rollout is old_rollout and pipeline.score_student_prefix is old_score


def test_aligned_control_restores_hooks_after_failure():
    import masked_ttt.pipeline as pipeline
    old_answer=pipeline.answer_standalone;old_decode=getattr(pipeline,'decode_standalone',None)
    with pytest.raises(RuntimeError,match='intentional'):
        with aligned_controls():raise RuntimeError('intentional')
    assert pipeline.answer_standalone is old_answer
    assert getattr(pipeline,'decode_standalone',None) is old_decode
