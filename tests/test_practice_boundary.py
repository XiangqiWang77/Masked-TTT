"""Recover a verified practice without changing its cached source prefix."""
from dataclasses import replace
import json

import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt.frozen_scores import layout_on_source_spans
from masked_ttt.selection import PreparedRequest
import masked_ttt.self_study_runtime as study
from test_batched_sd import ByteTokenizer


class BoundaryMergingTokenizer(ByteTokenizer):
    """Model a tokenizer with different final-dot/newline merge tokens."""
    def __call__(self,text,**kwargs):
        tokens=[];offsets=[];i=0
        while i<len(text):
            if text.startswith('.\n\n',i):token=300;width=3
            elif text.startswith('.\n',i):token=301;width=2
            else:token=ord(text[i]);width=1
            tokens.append(token);offsets.append((i,i+width));i+=width
        return dict(input_ids=tokens,offset_mapping=offsets)
    def encode(self,text,**kwargs):return self(text)['input_ids']
    def decode(self,tokens,**kwargs):
        return ''.join('.\n\n' if i==300 else '.\n' if i==301 else
                       chr(i) if 32<=i<127 else f'<{i}>' for i in tokens)


def test_original_whitespace_recovers_verified_question_without_teacher_answer_leak(monkeypatch):
    torch.manual_seed(173);torch.set_num_threads(1)
    config=Qwen3Config(vocab_size=512,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=2048,
        pad_token_id=0,eos_token_id=2)
    config._attn_implementation='eager'
    options=TTTConfig(chunk_tokens=64,max_context=2048,context_reserve_tokens=128,value_rank=2,
        steps=1,teacher_max_new_tokens=3,self_question_count=1,self_question_max_tokens=64,
        prefill_chunk_tokens=128,guard_enabled=False)
    engine=Backbone(Qwen3ForCausalLM(config).eval(),BoundaryMergingTokenizer(),options)
    document='Other notes. X final budget is 320.'
    question='How much did X finally receive?'
    request=Request(document=document,question=question,
        prompt='Context:\n'+document+'\nQuestion: '+question,max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,original):
            layout=layout_on_source_spans(backbone,original,[(0,len(original.document))])
            return PreparedRequest(original,layout,[0],dict(backend='synthetic'))
    proposal=json.dumps({'exercises':[dict(question='What is the final X budget?',answer='320',
                                            evidence='X final budget is 320.')]})
    monkeypatch.setattr(study,'generate_from_cache',lambda *args,**kwargs:([1],proposal))
    monkeypatch.setattr(study,'answer_quote',lambda *args,**kwargs:'320')
    baseline=MaskedTTT(engine,Selector()).answer(request)
    assert baseline['self_study']['verified']==0
    assert baseline['self_study']['checks'][0]['reason']=='document_prefix_mismatch'
    engine.config=replace(engine.config,practice_boundary_fallback=True)
    fixed=MaskedTTT(engine,Selector()).answer(request)
    assert fixed['self_study']['verified']==1
    assert fixed['self_study']['boundary_fallback_recovered']==1
    assert fixed['self_study']['checks'][0]['boundary_fallback_used']
    assert any(step['question_path']=='practice_1' for step in fixed['training']['steps'])
    assert not fixed['self_study']['reference_answers_used']
    assert not fixed['self_study']['student_answer_hint_used']
    assert not fixed['standalone_cartridge']['teacher_answer_in_final_prompt']
    assert fixed['backbone_unchanged'] and fixed['original_document_KV_unchanged']
