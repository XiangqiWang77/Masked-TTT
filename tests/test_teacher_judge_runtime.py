"""A verified Teacher contradiction prevents every request-local update."""
import json

import pytest
import torch
from transformers import (Qwen3Config,Qwen3ForCausalLM,Olmo3Config,Olmo3ForCausalLM,
                          GptOssConfig,GptOssForCausalLM)

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
import masked_ttt.pipeline as pipeline
import masked_ttt.teacher_judge_runtime as judge
from test_batched_sd import ByteTokenizer


@pytest.mark.parametrize('action,expected_steps',[('enforce',0),('observe',2)])
@pytest.mark.parametrize('architecture',['qwen3','olmo3','gpt_oss'])
def test_grounded_teacher_judgment_happens_before_any_update(monkeypatch,action,expected_steps,architecture):
    torch.manual_seed(97);torch.set_num_threads(1)
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
    options=TTTConfig(chunk_tokens=64,max_context=2048,context_reserve_tokens=128,
        value_rank=2,steps=2,teacher_max_new_tokens=3,self_question_count=2,
        teacher_judge_policy='grounded_veto',teacher_judge_action=action,teacher_judge_focused_tokens=128,
        teacher_judge_full_tokens=128,prefill_chunk_tokens=128,guard_enabled=True)
    model=model_type(model_config).eval()
    if architecture=='gpt_oss':
        for block in model.model.layers:
            with torch.no_grad():block.self_attn.sinks.fill_(.2)
    engine=Backbone(model,tokenizer,options)
    document='Other notes. '+('Filler. '*4)+'X final budget is 320. '+('More filler. '*4)
    question='How much is the final X budget?'
    request=Request(document=document,question=question,max_new_tokens=3)
    layout=prepare_layout(engine,request);evidence='X final budget is 320.'
    selected=[next(i for i,(a,b) in enumerate(layout.spans) if evidence in document[a:b])]
    class Selector:
        def prepare(self,backbone,original):
            return PreparedRequest(original,prepare_layout(backbone,original),selected,
                                   dict(backend='unit_test'))
    original_teacher=pipeline.build_teacher
    def teacher_with_candidate(*args):
        item=original_teacher(*args);item.complete=True;item.prediction='300';return item
    monkeypatch.setattr(pipeline,'build_teacher',teacher_with_candidate)
    calls=[]
    def assessment(*args,**kwargs):
        calls.append(engine.tokenizer.decode(args[4].tolist()))
        response=dict(reason='The source says the final amount is 320.',evidence=[evidence],
                      candidate_has_answer=True,candidate_contradicted=True)
        return [1,2],json.dumps(response)
    monkeypatch.setattr(judge,'generate_from_cache',assessment)
    def study(*args,**kwargs):
        if action=='enforce':raise AssertionError('Rejected teacher triggered self-study')
        return [],dict(verified=0,reference_answers_used=False,student_answer_hint_used=False)
    monkeypatch.setattr(pipeline,'generate_self_study',study)
    result=MaskedTTT(engine,Selector()).answer(request)
    assert len(calls)==2
    assert 'Prior source quotes to check against the current source' in calls[1]
    assert 'The source says the final amount is 320.' not in calls[1]
    assert result['teacher_judgment']['executed_before_optimizer']
    assert not result['teacher_judgment']['apply_update']
    assert result['teacher_judgment']['contradiction']
    assert result['teacher_judgment']['verified_focused_quotes']==[evidence]
    assert len(result['training']['steps'])==expected_steps
    assert result['teacher_judgment']['enforced']==(action=='enforce')
    if action=='enforce':
        assert result['support']['decision']['state']=='teacher_signal_rejected'
        assert result['prediction']==result['self_judge']['native_prediction']
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['final_question_recomputed'] and not result['teacher_answer_in_final_prompt']
    assert not result['teacher_judgment']['reference_answers_used']
