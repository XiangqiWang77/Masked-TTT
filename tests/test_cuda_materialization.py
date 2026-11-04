"""Check the real CUDA materialization against an independent attention formula."""
import math

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from masked_ttt.attention import address_attention


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Requires the real CUDA attention backend")
@pytest.mark.parametrize("query_heads,kv_heads,dim,has_sinks",[(32,8,128,False),(64,8,64,True)])
@pytest.mark.parametrize("bias_mode",["kv","query"])
def test_flash_materialization_matches_independent_formula_and_gradients(
    query_heads,kv_heads,dim,has_sinks,bias_mode
):
    torch.manual_seed(771)
    q=torch.randn(1,query_heads,3,dim,device="cuda",dtype=torch.bfloat16)
    k=torch.randn(1,kv_heads,17,dim,device="cuda",dtype=torch.bfloat16)
    value=torch.randn(1,kv_heads,17,dim,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    bias_heads=kv_heads if bias_mode=="kv" else query_heads
    bias=torch.randn(bias_heads,17,device="cuda",dtype=torch.float32,requires_grad=True)*.3
    sinks=(torch.randn(query_heads,device="cuda",dtype=torch.float32)*.2 if has_sinks else None)
    probe=torch.randn(1,query_heads,3,dim,device="cuda",dtype=torch.float32)
    observed=address_attention(q,k,value,bias,backend="flash",sinks=sinks)
    got_bias,got_value=torch.autograd.grad((observed.float()*probe).sum(),(bias,value))

    ref_bias=bias.detach().clone().requires_grad_()
    ref_value=value.detach().clone().requires_grad_()
    groups=query_heads//kv_heads
    scores=q.float()@repeat_kv(k.float(),groups).transpose(-1,-2)/math.sqrt(dim)
    head_bias=ref_bias.repeat_interleave(groups,0) if bias_mode=="kv" else ref_bias
    scores=scores+head_bias[None,:,None,:]
    visible=torch.arange(17,device="cuda")[None]<=torch.arange(14,17,device="cuda")[:,None]
    scores=scores.masked_fill(~visible[None,None],float("-inf"))
    if sinks is not None:
        scores=torch.cat((scores,sinks[None,:,None,None].expand(1,-1,3,-1)),-1)
    probabilities=scores.softmax(-1)
    if sinks is not None:probabilities=probabilities[...,:-1]
    expected=probabilities@repeat_kv(ref_value.float(),groups)
    want_bias,want_value=torch.autograd.grad((expected*probe).sum(),(ref_bias,ref_value))
    torch.testing.assert_close(observed.float(),expected,atol=.018,rtol=.04)
    torch.testing.assert_close(got_bias,want_bias,atol=.08,rtol=.08)
    torch.testing.assert_close(got_value.float(),want_value.float(),atol=.035,rtol=.05)
