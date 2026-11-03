"""GPT-OSS uses grouped query heads and learned attention sink logits."""
import math

import torch
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from masked_ttt import Backbone, MaskedTTT, Request, TTTConfig
from masked_ttt.attention import address_attention
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_batched_sd import ByteTokenizer


def test_grouped_bias_and_head_specific_sinks_match_direct_attention():
    torch.manual_seed(83);torch.set_num_threads(1)
    q=torch.randn(1,8,3,8,dtype=torch.float64,requires_grad=True)
    k=torch.randn(1,1,7,8,dtype=torch.float64,requires_grad=True)
    v=torch.randn(1,1,7,8,dtype=torch.float64,requires_grad=True)
    bias=torch.randn(1,7,dtype=torch.float64,requires_grad=True)
    sinks=torch.randn(8,dtype=torch.float64,requires_grad=True)
    keys=repeat_kv(k,8);values=repeat_kv(v,8)
    logits=q@keys.transpose(-1,-2)/math.sqrt(8)+bias.repeat_interleave(8,0)[None,:,None,:]
    visible=torch.arange(7)[None,:]<=torch.arange(4,7)[:,None]
    logits=logits.masked_fill(~visible[None,None],float('-inf'))
    sink=sinks[None,:,None,None].expand(1,8,3,1)
    expected=torch.cat((logits,sink),-1).softmax(-1)[...,:-1]@values
    actual=address_attention(q,k,v,bias,sinks=sinks,backend='math')
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    arguments=(q,k,v,bias,sinks)
    first=torch.autograd.grad(actual.square().sum(),arguments,retain_graph=True)
    second=torch.autograd.grad(expected.square().sum(),arguments)
    for left,right in zip(first,second):
        torch.testing.assert_close(left,right,atol=1e-11,rtol=1e-10)


def test_gpt_oss_eight_queries_per_kv_head_trains_both_paths():
    torch.manual_seed(89);torch.set_num_threads(1)
    config=GptOssConfig(vocab_size=512,hidden_size=64,intermediate_size=96,
        num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=1024,layer_types=['sliding_attention','full_attention'],
        sliding_window=16,pad_token_id=0,eos_token_id=2,num_local_experts=2,
        num_experts_per_tok=1,rope_scaling=None)
    config._attn_implementation='eager'
    model=GptOssForCausalLM(config).eval()
    for block in model.model.layers:
        with torch.no_grad():block.self_attn.sinks.copy_(torch.linspace(-.3,.3,8))
    class HarmonyTokenizer(ByteTokenizer):
        def apply_chat_template(self,messages,**kwargs):
            return '<|start|>user<|message|>'+messages[0]['content']+'<|end|><|start|>assistant'
    options=TTTConfig(chunk_tokens=32,max_context=1024,context_reserve_tokens=128,
        value_rank=2,steps=2,teacher_max_new_tokens=3,self_question_count=0,
        prefill_chunk_tokens=64,guard_enabled=False,matched_cache_direct=True)
    engine=Backbone(model,HarmonyTokenizer(),options)
    request=Request(document='The old note is red. The final note is blue. '*4,
                    question='What color is the final note?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,original):
            layout=prepare_layout(backbone,original)
            return PreparedRequest(original,layout,[len(layout.chunks)//2],dict(backend='synthetic'))
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result['training']['accepted_steps']==2
    assert result['training']['adapted_layers']==[1]
    assert all(step['gain_gradient_norm']>0 and step['value_gradient_norm']>0
               for step in result['training']['steps'])
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['matched_cache_study']['source_selected_cache_unchanged']
