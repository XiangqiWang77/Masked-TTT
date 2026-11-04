from dataclasses import replace

import pytest
import torch

from masked_ttt import MaskedTTT,Request,TTTConfig
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine,setup_views
from test_standalone import make_engine


@pytest.mark.parametrize("architecture",["qwen3","gpt_oss"])
def test_selected_basis_uses_independent_selected_values_and_shares_fixed_buffer(architecture):
    engine=make_engine() if architecture=="qwen3" else gpt_engine()
    _,_,_,_,teacher,full,view=setup_views(engine)
    before=[(x.values._version,x.values.clone()) for x in teacher.document_cache.layers]
    view.memory.prepare_value_basis(teacher.document_cache,full.adapted_layers)
    assert view.memory.value_basis is full.value_basis
    assert not full.value_basis.requires_grad
    for layer in full.adapted_layers:
        native=teacher.document_cache.layers[layer].values
        x=native[0,:,view.memory.selected_positions].float()
        covariance=x.transpose(-1,-2)@x/max(1,x.shape[-2])
        eigenvalues,_=torch.linalg.eigh(covariance)
        captured=(x@full.value_basis[layer]).square().sum(dim=(-2,-1))
        maximum=eigenvalues[...,-full.value_basis.shape[-1]:].sum(-1)*x.shape[-2]
        torch.testing.assert_close(captured,maximum,rtol=1e-4,atol=1e-5)
    for item,(version,saved) in zip(teacher.document_cache.layers,before):
        assert item.values._version==version
        torch.testing.assert_close(item.values,saved,atol=0,rtol=0)


@pytest.mark.parametrize("architecture",["qwen3","gpt_oss"])
def test_selected_basis_preserves_full_student_sd_and_standalone_release(architecture):
    engine=make_engine() if architecture=="qwen3" else gpt_engine()
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=0.,value_basis_source="selected",
        release_full_before_cartridge=True,standalone_cartridge_enabled=True)
    document="Old report says red. New report says blue. "*6
    request=Request(document=document,question="What is the new report?",max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend="unit"))
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result["training"]["objective"]=="on-policy KL(teacher || student)"
    assert result["training"]["selected_transfer_weight"]==0
    assert result["training"]["value_basis_source"]=="selected"
    assert any(x["gain_gradient_norm"]>0 for x in result["training"]["steps"])
    assert any(x["value_gradient_norm"]>0 for x in result["training"]["steps"])
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"]==0
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    assert not result["teacher_answer_in_final_prompt"]


def test_invalid_basis_source_is_rejected():
    with pytest.raises(ValueError):TTTConfig(value_basis_source="unrelated")
