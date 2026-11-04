"""Self-questions must share the same immutable document and trainable memory."""
import json

import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
import masked_ttt.pipeline as pipeline
import masked_ttt.self_study_runtime as study
from test_batched_sd import ByteTokenizer


def test_verified_subquestion_joins_main_opd_on_the_same_memory(monkeypatch):
    torch.manual_seed(91);torch.set_num_threads(1)
    model_config=Qwen3Config(vocab_size=512,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=2048,pad_token_id=0,eos_token_id=2)
    model_config._attn_implementation='eager'
    config=TTTConfig(chunk_tokens=64,max_context=2048,context_reserve_tokens=128,
        value_rank=2,steps=2,teacher_max_new_tokens=3,self_question_count=2,
        self_question_max_tokens=64,prefill_chunk_tokens=128,guard_enabled=False)
    engine=Backbone(Qwen3ForCausalLM(model_config).eval(),ByteTokenizer(),config)
    document='Other project notes. '+('Filler statement. '*4)+'X final budget is 320. '+('More filler. '*4)
    question='How much is the final X budget?'
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
    monkeypatch.setattr(study,'answer_quote',lambda *args,**kwargs:'320')
    memories=[];original_logits=pipeline.batched_teacher_logits
    def observe(engine,reader,*args,**kwargs):
        memories.append(id(reader.memory));return original_logits(engine,reader,*args,**kwargs)
    monkeypatch.setattr(pipeline,'batched_teacher_logits',observe)
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result['self_study']['verified']==1
    assert [s['question_path'] for s in result['training']['steps']]==['main','practice_1','main']
    assert len(set(memories))==1
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['final_question_recomputed'] and not result['teacher_answer_in_final_prompt']
    assert result['self_study']['reference_answers_used'] is False
    assert result['standalone_cartridge']['memory']['additional_source_kv_bytes_before_decode']==0
