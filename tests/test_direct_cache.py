"""The standalone fast path must reproduce the paired selected-cache study."""
from dataclasses import replace
import json

import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM

from masked_ttt import Backbone, MaskedTTT, Request, TTTConfig
from masked_ttt.memory import KVReadMemory
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_batched_sd import ByteTokenizer
from test_standalone import make_engine


class Selector:
    def prepare(self, engine, request):
        layout=prepare_layout(engine,request)
        return PreparedRequest(request,layout,[len(layout.chunks)//2],dict(backend='synthetic'))


def gpt_engine():
    torch.manual_seed(71);torch.set_num_threads(1)
    config=GptOssConfig(vocab_size=512,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
        layer_types=['sliding_attention','full_attention'],sliding_window=16,pad_token_id=0,
        eos_token_id=2,head_dim=8,num_local_experts=2,num_experts_per_tok=1,rope_scaling=None)
    config._attn_implementation='eager'
    model=GptOssForCausalLM(config).eval()
    for block in model.model.layers:
        with torch.no_grad():block.self_attn.sinks.fill_(.2)
    class HarmonyTokenizer(ByteTokenizer):
        def apply_chat_template(self,messages,**kwargs):
            return '<|start|>user<|message|>'+messages[0]['content']+'<|end|><|start|>assistant'
    options=TTTConfig(chunk_tokens=32,max_context=1024,context_reserve_tokens=128,value_rank=2,
        steps=2,teacher_max_new_tokens=3,self_question_count=0,prefill_chunk_tokens=64,
        guard_enabled=False,standalone_cartridge_enabled=True)
    return Backbone(model,HarmonyTokenizer(),options)


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_direct_path_matches_all_paired_predictions_and_gradients(architecture,monkeypatch):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    request=Request(document='Old report says red. New report says blue. '*6,
                    question='What is the new report?',max_new_tokens=3)
    engine.config=replace(engine.config,matched_cache_probe=True)
    reference=MaskedTTT(engine,Selector()).answer(request)['matched_cache_study']
    calls=[];original=KVReadMemory.prepare_value_basis
    def record(memory,cache,layers):
        calls.append(cache.layers[memory.adapted_layers[0]].values.shape[-2])
        return original(memory,cache,layers)
    monkeypatch.setattr(KVReadMemory,'prepare_value_basis',record)
    engine.config=replace(engine.config,matched_cache_probe=False,matched_cache_direct=True)
    direct=MaskedTTT(engine,Selector()).answer(request)
    assert len(calls)==1 and calls[0]==reference['selected_document_tokens']
    assert direct['training']['training_cache']=='selected_document'
    assert direct['standalone_cartridge']['generated_token_ids']==reference['trained']['generated_token_ids']
    for name in ('zero_update','trained','zero_format','trained_format'):
        assert direct['matched_cache_study'][name]['generated_token_ids']==reference[name]['generated_token_ids']
    for a,b in zip(direct['matched_cache_study']['steps'],reference['steps']):
        assert a['student_rollout_tokens']==b['student_rollout_tokens']
        assert a['target_positions']==b['target_positions']
        for key in ('teacher_KL_before','gain_gradient_norm','value_gradient_norm'):
            assert a[key]==pytest.approx(b[key],rel=1e-5,abs=1e-7)
    assert len(direct['matched_cache_study']['steps'])==len(reference['steps'])
    assert direct['original_document_KV_unchanged'] and direct['backbone_unchanged']
    assert not direct['standalone_cartridge']['teacher_answer_in_final_prompt']
    assert not direct['standalone_cartridge']['memory']['raw_full_document_kv_used_at_decode']
    engine.config=replace(engine.config,matched_cache_final_suffix='teacher_format')
    formatted=MaskedTTT(engine,Selector()).answer(request)
    assert formatted['standalone_cartridge']['generated_token_ids']==reference['trained_format']['generated_token_ids']
    assert not formatted['standalone_cartridge']['teacher_answer_in_final_prompt']
    engine.config=replace(engine.config,matched_cache_final_suffix='practice_gated_format')
    gated=MaskedTTT(engine,Selector()).answer(request)
    assert gated['standalone_cartridge']['selected_output_arm']=='zero_format'
    assert gated['standalone_cartridge']['generated_token_ids']==reference['zero_format']['generated_token_ids']
    assert not gated['standalone_cartridge']['teacher_answer_in_final_prompt']


def test_direct_configuration_rejects_ambiguous_modes():
    with pytest.raises(ValueError,match='requires standalone'):
        TTTConfig(matched_cache_direct=True,standalone_cartridge_enabled=False)
    with pytest.raises(ValueError,match='without the paired probe'):
        TTTConfig(matched_cache_direct=True,matched_cache_probe=True)
    with pytest.raises(ValueError,match='requires direct'):
        TTTConfig(matched_cache_final_suffix='teacher_format')
    with pytest.raises(ValueError,match='requires direct'):
        TTTConfig(matched_cache_final_suffix='practice_gated_format')


def test_direct_path_preserves_verified_practice_when_main_teacher_is_rejected(monkeypatch):
    import masked_ttt.pipeline as pipeline
    import masked_ttt.self_study_runtime as study
    engine=make_engine()
    engine.config=replace(engine.config,chunk_tokens=64,self_question_count=1,
        teacher_judge_policy='grounded_veto',practice_when_teacher_rejected=True,
        matched_cache_probe=True)
    evidence='X final budget is 320.'
    document='Other notes. '+'Filler. '*2+evidence+' More filler. '*10
    request=Request(document=document,question='How much did X receive after final approval?',max_new_tokens=3)
    layout=prepare_layout(engine,request)
    selected=[next(i for i,(a,b) in enumerate(layout.spans) if evidence in document[a:b])]
    class EvidenceSelector:
        def prepare(self,backbone,original):
            return PreparedRequest(original,prepare_layout(backbone,original),selected,dict(backend='synthetic'))
    proposal=json.dumps({'exercises':[dict(question='What is the final X budget?',answer='320',evidence=evidence)]})
    monkeypatch.setattr(study,'generate_from_cache',lambda *args,**kwargs:([1],proposal))
    monkeypatch.setattr(study,'answer_quote',lambda *args,**kwargs:'320')
    monkeypatch.setattr(pipeline,'judge_teacher',lambda *args,**kwargs:dict(
        apply_update=False,reference_answers_used=False,reason='grounded_contradiction',seconds=0.))
    reference=MaskedTTT(engine,EvidenceSelector()).answer(request)
    engine.config=replace(engine.config,matched_cache_probe=False,matched_cache_direct=True)
    direct=MaskedTTT(engine,EvidenceSelector()).answer(request)
    for result in (reference,direct):
        assert result['self_study']['verified']==1
        assert result['support']['decision']['state']=='verified_practice_only'
        assert not result['support']['decision']['teacher_main_signal_used']
        assert not result['self_study']['reference_answers_used']
    expected=reference['matched_cache_study'];actual=direct['matched_cache_study']
    assert [step['question_path'] for step in actual['steps']]==['practice_1','practice_1']
    assert actual['trained']['generated_token_ids']==expected['trained']['generated_token_ids']
    assert actual['trained_format']['generated_token_ids']==expected['trained_format']['generated_token_ids']
    for a,b in zip(actual['steps'],expected['steps']):
        assert a['student_rollout_tokens']==b['student_rollout_tokens']
        assert a['teacher_KL_before']==pytest.approx(b['teacher_KL_before'],rel=1e-5,abs=1e-7)
    engine.config=replace(engine.config,matched_cache_final_suffix='practice_gated_format')
    gated=MaskedTTT(engine,EvidenceSelector()).answer(request)
    assert gated['standalone_cartridge']['selected_output_arm']=='trained_format'
    assert gated['standalone_cartridge']['generated_token_ids']==expected['trained_format']['generated_token_ids']
    assert not gated['standalone_cartridge']['teacher_answer_in_final_prompt']
