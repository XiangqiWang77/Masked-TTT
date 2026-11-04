from dataclasses import replace

import pytest
import torch
from transformers import Qwen3Config

from experiments.precision_cartridge.materialization import (
    _float_values_during_materialization,_mixed_precision_attention,float32_cartridge,
)
from masked_ttt import MaskedTTT,Request
from masked_ttt.memory import KVReadMemory
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def test_float32_value_add_preserves_a_small_trained_residual():
    torch.manual_seed(8)
    config=Qwen3Config(hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
                       num_key_value_heads=1,head_dim=8)
    memory=KVReadMemory(config,[list(range(1,17))],[0],17,rank=2,sink_tokens=1)
    memory.value_enabled=True;memory.active_value_layers={0}
    with torch.no_grad():memory.value_coefficients[0].normal_(std=.0001)
    native=torch.randn(1,1,17,8,dtype=torch.bfloat16)
    x=native.float();basis=memory.value_basis[0,0]
    coefficient=memory.value_coefficients[0,0,0]
    expected=x.clone();expected[:,:,1:,:]+=(x[:,:,1:,:]@basis)@coefficient
    precise=memory.values(0,native.float())
    rounded=memory.values(0,native).float()
    assert precise.dtype==torch.float32
    assert torch.allclose(precise,expected,atol=1e-7,rtol=1e-7)
    assert (rounded-expected).square().sum()>(precise-expected).square().sum()


@pytest.mark.parametrize('has_sinks',[False,True])
def test_mixed_value_attention_matches_causal_gqa_float32(has_sinks):
    import masked_ttt.architectures as architecture
    import masked_ttt.reader as reader
    torch.manual_seed(9)
    q=torch.randn(1,4,3,8,dtype=torch.bfloat16)
    k=torch.randn(1,2,7,8,dtype=torch.bfloat16)
    v=torch.randn(1,2,7,8,dtype=torch.float32)
    options=dict(scale=.5,sinks=torch.randn(4) if has_sinks else None)
    original=reader.native_attention
    expanded_k=k.float().repeat_interleave(2,dim=1)
    expanded_v=v.repeat_interleave(2,dim=1)
    logits=(q.float()@expanded_k.transpose(-1,-2))*.5
    allowed=torch.arange(7)[None,:] <= (torch.arange(3)[:,None]+4)
    logits=logits.masked_fill(~allowed[None,None],float('-inf'))
    if has_sinks:
        logits=torch.cat((logits,options['sinks'][None,:,None,None].expand(1,4,3,1)),dim=-1)
    weights=logits.softmax(-1)[...,:7]
    expected=(weights@expanded_v).to(q.dtype)
    with _mixed_precision_attention():
        actual=reader.native_attention(q,k,v,**options)
        hybrid=architecture.native_attention(q,k,v,**options)
    assert actual.dtype==q.dtype and torch.equal(actual,expected)
    assert torch.equal(hybrid,expected)
    assert reader.native_attention is original


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_standalone_float32_cache_preserves_original_training_and_releases_sources(architecture):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.model.to(dtype=torch.bfloat16)
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=0.,release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request=Request(document='Old note is red. Current note is blue. '*6,
                    question='Which color is current?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    method=MaskedTTT(engine,Selector());baseline=method.answer(request)
    with float32_cartridge(matched_controls=True):result=method.answer(request)
    memory=result['standalone_cartridge']['memory']
    assert memory['experimental_materialization']=='float32_value_cache'
    assert 'torch.float32' in memory['experimental_value_dtypes']
    assert memory['experimental_float_attention_calls']>0
    controls=result['standalone_cartridge']['precision_matched_controls']
    assert controls['adapted']['tokens']==result['standalone_cartridge']['generated_token_ids']
    assert controls['same_precision_backend'] and controls['same_selected_kv']
    assert not controls['teacher_answer_in_student_prompt'] and not controls['reference_answers_used']
    assert memory['additional_source_kv_bytes_before_decode']==0
    assert memory['additional_training_adapter_bytes_before_decode']==0
    assert result['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']==0
    assert result['training']['objective']=='on-policy KL(teacher || student)'
    assert result['training']['selected_transfer_weight']==0 and result['training']['steps']
    assert result['teacher']['tokens']==baseline['teacher']['tokens']
    assert result['self_judge']['adapted_prediction']==baseline['self_judge']['adapted_prediction']
    for first,second in zip(result['training']['steps'],baseline['training']['steps']):
        assert first['student_rollout_tokens']==second['student_rollout_tokens']
        assert first['teacher_KL_before']==pytest.approx(second['teacher_KL_before'],abs=1e-8)
        assert first['gain_gradient_norm']==pytest.approx(second['gain_gradient_norm'],abs=1e-8)
        assert first['value_gradient_norm']==pytest.approx(second['value_gradient_norm'],abs=1e-8)
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert not result['teacher_answer_in_final_prompt'] and result['generated_token_ids']


def test_zero_update_keeps_native_value_cache_and_prediction():
    engine=make_engine();engine.model.to(dtype=torch.bfloat16)
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,steps=0,
        selected_transfer_weight=0.,release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request=Request(document='Old note is red. Current note is blue. '*6,
                    question='Which color is current?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    method=MaskedTTT(engine,Selector());baseline=method.answer(request)
    with float32_cartridge():result=method.answer(request)
    assert result['standalone_cartridge']['generated_token_ids']==baseline['standalone_cartridge']['generated_token_ids']
    assert result['standalone_cartridge']['memory']['experimental_materialization']=='native_no_active_value_update'


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_loaded_backbone_probe_is_paired_and_restores_config(architecture):
    from experiments.precision_cartridge.real_model_probe import validate_loaded_backbone
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.model.to(dtype=torch.bfloat16)
    previous=engine.config
    probe=validate_loaded_backbone(engine)
    assert probe['passed'] and probe['same_teacher_and_training']
    assert probe['full_and_source_state_released'] and not probe['reference_answers_used']
    assert engine.config is previous


def test_experimental_hooks_restore_after_failure():
    import masked_ttt.architectures as architecture
    import masked_ttt.pipeline as pipeline
    import masked_ttt.reader as reader
    answer=pipeline.answer_standalone;decode=pipeline.decode_standalone
    with pytest.raises(RuntimeError,match='intentional'):
        with float32_cartridge():raise RuntimeError('intentional')
    assert pipeline.answer_standalone is answer and pipeline.decode_standalone is decode
    compile_branch=reader.MemoryReader.compiled_branch
    native=reader.native_attention;hybrid=architecture.native_attention
    with pytest.raises(RuntimeError,match='intentional'):
        with _float_values_during_materialization(),_mixed_precision_attention():
            raise RuntimeError('intentional')
    assert reader.MemoryReader.compiled_branch is compile_branch
    assert reader.native_attention is native and architecture.native_attention is hybrid
