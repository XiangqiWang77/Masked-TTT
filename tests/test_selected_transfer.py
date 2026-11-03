from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers import GptOssConfig,GptOssForCausalLM

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt.memory import KVReadMemory
from masked_ttt.on_policy import student_rollout
from masked_ttt.reader import MemoryReader
from masked_ttt.request import prepare_layout
from masked_ttt.runtime import private_module_tree
from masked_ttt.selection import PreparedRequest
from masked_ttt.shared_selected import shared_selected_reader
from masked_ttt.standalone import build_standalone
from masked_ttt.teacher import build_teacher,prefill_document
from test_batched_sd import ByteTokenizer
from test_standalone import make_engine


def setup_views(engine):
    document="First fact is red. "+"Unrelated material. "*8+"Second fact is blue. "+"More noise. "*8
    request=Request(document=document,question="Which color is second?",max_new_tokens=3)
    layout=prepare_layout(engine,request)
    selected=[1,len(layout.chunks)-2]
    assert selected==sorted(set(selected))
    reader=SimpleNamespace(model=private_module_tree(engine.model.model),lm_head=engine.model.lm_head)
    full_cache=prefill_document(engine,engine.model,layout.ids,layout.initial,
                                [c[-1]+1 for c in layout.chunks])
    teacher=build_teacher(engine,reader,request,layout,selected)
    full=KVReadMemory(engine.model.config,layout.chunks,selected,layout.document_end,
        rank=engine.config.value_rank,value_radius=engine.config.value_radius,
        gain_bound=engine.config.gain_bound,sink_tokens=engine.config.sink_tokens).to(engine.device)
    full.prepare_value_basis(full_cache,full.adapted_layers)
    full.value_enabled=True
    view=shared_selected_reader(engine,layout,selected,teacher,full)
    return request,layout,selected,full_cache,teacher,full,view


def test_selected_view_shares_parameters_and_matches_final_cartridge():
    engine=make_engine()
    request,layout,selected,_,teacher,full,view=setup_views(engine)
    assert view.memory.gain is full.gain
    assert view.memory.value_coefficients is full.value_coefficients
    with torch.no_grad():
        full.gain[...,selected[0]]=0.2
        full.gain[...,selected[1]]=-0.3
        full.value_coefficients.normal_(0,.01)
    baseline=[x.values.clone() for x in teacher.document_cache.layers]
    standalone,_,_,_,_=build_standalone(engine,layout,selected,teacher,full)
    for layer in full.adapted_layers:
        native=teacher.document_cache.layers[layer].values
        assert torch.allclose(view.memory.biases(layer,native.shape[-2]),
                              standalone.memory.biases(layer,native.shape[-2]))
        assert torch.allclose(view.memory.values(layer,native),standalone.memory.values(layer,native))
    suffix=torch.tensor(layout.ids[layout.document_end:],device=engine.device)
    assert torch.equal(student_rollout(engine,view,teacher.document_cache,suffix,3),
                       student_rollout(engine,standalone,teacher.document_cache,suffix,3))
    loss=0
    for layer in full.adapted_layers:
        native=teacher.document_cache.layers[layer].values
        loss=loss+view.memory.biases(layer,native.shape[-2]).sum()+view.memory.values(layer,native).square().sum()
    loss.backward()
    assert full.gain.grad is not None and full.gain.grad[...,selected].abs().sum()>0
    assert not full.gain.grad[...,~full.selected_chunk_mask].any()
    assert full.value_coefficients.grad is not None and full.value_coefficients.grad.abs().sum()>0
    assert all(torch.equal(x.values,y) for x,y in zip(teacher.document_cache.layers,baseline))


def test_transfer_targets_full_view_without_gradient(monkeypatch):
    import masked_ttt.selected_transfer as transfer
    engine=make_engine()
    request,layout,selected,cache,teacher,full,view=setup_views(engine)
    full_reader=MemoryReader(engine,full)
    suffix=torch.tensor(layout.ids[layout.document_end:],device=engine.device)
    calls=[];original=transfer.batched_teacher_logits
    def record(*args,**kwargs):
        calls.append(torch.is_grad_enabled())
        return original(*args,**kwargs)
    monkeypatch.setattr(transfer,"batched_teacher_logits",record)
    loss,rollout,positions=transfer.selected_transfer_loss(
        engine,full_reader,view,cache,teacher.document_cache,suffix,3)
    assert calls==[False,True]
    assert loss.requires_grad and len(rollout)>0 and positions
    loss.backward()
    assert full.gain.grad is not None and full.value_coefficients.grad is not None


def test_transfer_loss_stops_gradient_through_full_target(monkeypatch):
    import masked_ttt.selected_transfer as transfer
    full_parameter=torch.nn.Parameter(torch.tensor([[2.0,-1.0]]))
    selected_parameter=torch.nn.Parameter(torch.tensor([[0.0,0.0]]))
    full_reader=object();selected_reader=object()
    monkeypatch.setattr(transfer,"student_rollout",lambda *a,**k:torch.tensor([4]))
    def logits(_engine,reader,*args,**kwargs):
        # Explicitly enable autograd here to test the detach even if a helper
        # accidentally returns a tensor connected to the full-view graph.
        with torch.enable_grad():
            return full_parameter*1 if reader is full_reader else selected_parameter*1
    monkeypatch.setattr(transfer,"batched_teacher_logits",logits)
    loss,_,_=transfer.selected_transfer_loss(SimpleNamespace(),full_reader,selected_reader,
                                              None,None,torch.tensor([3]),1)
    loss.backward()
    assert full_parameter.grad is None
    assert selected_parameter.grad is not None and selected_parameter.grad.abs().sum()>0


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


@pytest.mark.parametrize("architecture",["qwen3","gpt_oss"])
def test_full_request_uses_shared_transfer_and_preserves_frozen_state(architecture):
    engine=make_engine() if architecture=="qwen3" else gpt_engine()
    engine.config=replace(engine.config,selected_transfer_weight=.25,self_question_count=0)
    request=Request(document="Old report says red. New report says blue. "*6,
                    question="What is the new report?",max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,original):
            layout=prepare_layout(backbone,original)
            return PreparedRequest(original,layout,[1,len(layout.chunks)-2],dict(backend="unit_test"))
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result["training"]["selected_transfer_weight"]==.25
    assert result["training"]["selected_transfer_full_target_detached"]
    assert result["training"]["selected_transfer_uses_selected_student_prefix"]
    assert all("selected_transfer_KL_before" in step for step in result["training"]["steps"])
    assert any(step["gain_gradient_norm"]>0 for step in result["training"]["steps"])
    assert any(step["value_gradient_norm"]>0 for step in result["training"]["steps"])
    assert result["original_document_KV_unchanged"] and result["backbone_unchanged"]
    assert not result["teacher_answer_in_final_prompt"]


def test_invalid_transfer_weight_is_rejected():
    with pytest.raises(ValueError):TTTConfig(selected_transfer_weight=-1)
    with pytest.raises(ValueError):TTTConfig(selected_transfer_weight=float("nan"))
