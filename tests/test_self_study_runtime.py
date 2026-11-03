"""Self-questions must share the same immutable document and trainable memory."""
import json

import pytest
import torch
from transformers import (Qwen3Config,Qwen3ForCausalLM,Olmo3Config,Olmo3ForCausalLM,
                          GptOssConfig,GptOssForCausalLM)

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
import masked_ttt.pipeline as pipeline
import masked_ttt.self_study_runtime as study
from test_batched_sd import ByteTokenizer


@pytest.mark.parametrize('architecture',['qwen3','olmo3','gpt_oss'])
@pytest.mark.parametrize('reject_main,verified,compare_native',[
    (False,True,False),(True,True,False),(True,False,False),(True,True,True)])
def test_verified_subquestion_joins_main_opd_on_the_same_memory(monkeypatch,architecture,reject_main,verified,compare_native):
    torch.manual_seed(91);torch.set_num_threads(1)
    common=dict(vocab_size=512,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,max_position_embeddings=2048,pad_token_id=0,eos_token_id=2)
    tokenizer=ByteTokenizer()
    if architecture=='qwen3':
        model_config=Qwen3Config(**common,num_key_value_heads=1,head_dim=8)
        model_type=Qwen3ForCausalLM
    elif architecture=='olmo3':
        model_config=Olmo3Config(**common,num_key_value_heads=2,
            layer_types=['sliding_attention','full_attention'],sliding_window=16)
        model_type=Olmo3ForCausalLM
    else:
        model_config=GptOssConfig(**common,num_key_value_heads=2,head_dim=8,
            layer_types=['sliding_attention','full_attention'],sliding_window=16,
            num_local_experts=2,num_experts_per_tok=1,rope_scaling=None)
        model_type=GptOssForCausalLM
        class HarmonyByteTokenizer(ByteTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                return 'USER:\n'+messages[0]['content']+'\n<|start|>assistant'
        tokenizer=HarmonyByteTokenizer()
    model_config._attn_implementation='eager'
    config=TTTConfig(chunk_tokens=64,max_context=2048,context_reserve_tokens=128,
        value_rank=2,steps=2,teacher_max_new_tokens=3,self_question_count=2,
        self_question_max_tokens=64,prefill_chunk_tokens=128,guard_enabled=compare_native,
        teacher_judge_policy='grounded_veto' if reject_main and not compare_native else 'off',
        practice_when_teacher_rejected=reject_main,
        teacher_veto_against_native=compare_native,
        adaptive_open_ended=reject_main,max_answer_steps=5)
    model=model_type(model_config).eval()
    if architecture=='gpt_oss':
        for block in model.model.layers:
            with torch.no_grad():block.self_attn.sinks.fill_(.2)
    engine=Backbone(model,tokenizer,config)
    document='Other project notes. '+('Filler statement. '*4)+'X final budget is 320. '+('More filler. '*4)
    question=('How much is the final X budget? Choices: A. 250 B. 320'
              if reject_main else 'How much is the final X budget?')
    request=Request(document=document,question=question,max_new_tokens=3)
    layout=prepare_layout(engine,request);evidence='X final budget is 320.'
    selected=[next(i for i,(a,b) in enumerate(layout.spans) if evidence in document[a:b])]
    class Selector:
        def prepare(self,backbone,original):
            return PreparedRequest(original,prepare_layout(backbone,original),selected,
                                   dict(backend='unit_test'))
    generated=json.dumps({'exercises':[dict(question='What is the final X budget?',answer='320',
                                             evidence=evidence)]})
    monkeypatch.setattr(study,'generate_from_cache',lambda *args,**kwargs:([1],generated))
    monkeypatch.setattr(study,'answer_quote',lambda *args,**kwargs:'320' if verified else '250')
    if reject_main and not compare_native:
        # The main answer is vetoed. The source-grounded question generator is
        # independent of that answer and still has to pass the normal checks.
        monkeypatch.setattr(pipeline,'judge_teacher',lambda *args,**kwargs:dict(
            apply_update=False,reference_answers_used=False,reason='grounded_contradiction',seconds=0.))
    if compare_native:
        monkeypatch.setattr(pipeline,'guard_answers',lambda *args,**kwargs:dict(
            use_adapted=False,seconds=0.,reason='native_preferred',reference_answers_used=False))
    memories=[];original_logits=pipeline.batched_teacher_logits
    def observe(engine,reader,*args,**kwargs):
        memories.append(id(reader.memory));return original_logits(engine,reader,*args,**kwargs)
    monkeypatch.setattr(pipeline,'batched_teacher_logits',observe)
    teacher_suffixes=[];student_suffixes=[]
    original_teacher=pipeline.score_student_prefix;original_rollout=pipeline.student_rollout
    def observe_teacher(engine,reader,teacher,rollout):
        teacher_suffixes.append(engine.tokenizer.decode(teacher.question_tokens.tolist()))
        return original_teacher(engine,reader,teacher,rollout)
    def observe_rollout(engine,reader,cache,suffix,limit):
        student_suffixes.append(engine.tokenizer.decode(suffix.tolist()))
        return original_rollout(engine,reader,cache,suffix,limit)
    monkeypatch.setattr(pipeline,'score_student_prefix',observe_teacher)
    monkeypatch.setattr(pipeline,'student_rollout',observe_rollout)
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result['self_study']['verified']==int(verified)
    paths=[s['question_path'] for s in result['training']['steps']]
    expected=(['practice_1','practice_1'] if reject_main and verified else
              [] if reject_main else ['main','practice_1','main'])
    assert paths==expected
    assert len(set(memories))==int(verified)
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['final_question_recomputed'] and not result['teacher_answer_in_final_prompt']
    assert result['self_study']['reference_answers_used'] is False
    assert result['self_study']['teacher_answer_hint_used'] is verified
    assert any('source-supported short answer is: 320' in x for x in teacher_suffixes) is verified
    assert all('source-supported short answer' not in x for x in student_suffixes)
    assert result['training']['teacher_main_signal_used'] is (not reject_main)
    if reject_main and verified:
        assert result['support']['decision']['state']=='verified_practice_only'
        assert all('source-supported short answer is: 320' in x for x in teacher_suffixes)
    if reject_main and not verified:
        assert result['support']['decision']['state']=='teacher_signal_rejected'
        assert teacher_suffixes==[] and student_suffixes==[]
