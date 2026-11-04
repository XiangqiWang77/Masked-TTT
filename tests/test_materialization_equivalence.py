"""A materialized cartridge must match applying the same read state dynamically."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers import GptOssConfig,GptOssForCausalLM

from masked_ttt import Backbone,Request,TTTConfig
from masked_ttt.cache import fork_prefix
from masked_ttt.memory import KVReadMemory
from masked_ttt.request import prepare_layout
from masked_ttt.runtime import private_forward,private_module_tree
from masked_ttt.standalone import build_standalone
from masked_ttt.teacher import build_teacher,prefill_document
from test_batched_sd import ByteTokenizer
from masked_ttt.model import Budget
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def gpt_gqa_engine():
    torch.manual_seed(71);torch.set_num_threads(1)
    config=GptOssConfig(vocab_size=512,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024,
        layer_types=['sliding_attention','full_attention'],sliding_window=16,pad_token_id=0,
        eos_token_id=2,head_dim=8,num_local_experts=2,num_experts_per_tok=1,rope_scaling=None)
    config._attn_implementation='eager'
    model=GptOssForCausalLM(config).eval()
    for block in model.model.layers:
        with torch.no_grad():block.self_attn.sinks.fill_(.2)
    class HarmonyTokenizer(ByteTokenizer):
        def apply_chat_template(self,messages,**kwargs):
            return '<|start|>user<|message|>'+messages[0]['content']+'<|end|><|start|>assistant'
    return Backbone(model,HarmonyTokenizer(),TTTConfig(chunk_tokens=32,max_context=1024,
        context_reserve_tokens=128,value_rank=2,steps=2,teacher_max_new_tokens=3,
        self_question_count=0,prefill_chunk_tokens=64,guard_enabled=False,
        standalone_cartridge_enabled=True))


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='Requires CUDA attention backend'))])
@pytest.mark.parametrize('architecture',['qwen3','gpt_oss','gpt_oss_gqa'])
def test_materialized_selected_kv_matches_dynamic_read_and_gradients(architecture,device):
    engine=(make_engine() if architecture=='qwen3' else
            gpt_engine() if architecture=='gpt_oss' else gpt_gqa_engine())
    if device=='cuda':
        engine.model.to(device='cuda:0',dtype=torch.bfloat16)
        engine.device=torch.device('cuda:0')
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=0.,value_basis_source='document')
    engine.budget=Budget(None)
    request=Request(document='Old note is red. Current note is blue. '*6,
                    question='Which color is current?',max_new_tokens=3)
    layout=prepare_layout(engine,request);selected=[1,len(layout.chunks)-2]
    full=prefill_document(engine,engine.model,layout.ids,layout.initial,
                          [chunk[-1]+1 for chunk in layout.chunks])
    teacher_reader=SimpleNamespace(model=private_module_tree(engine.model.model),lm_head=engine.model.lm_head)
    teacher=build_teacher(engine,teacher_reader,request,layout,selected)
    trained=KVReadMemory(engine.model.config,layout.chunks,selected,layout.document_end,
        rank=2,sink_tokens=engine.config.sink_tokens,seed=engine.config.seed).to(engine.device)
    trained.prepare_value_basis(full,trained.adapted_layers)
    trained.value_enabled=True;trained.native_mode=False
    torch.manual_seed(311)
    with torch.no_grad():
        for layer in trained.adapted_layers:
            trained.gain[layer,:,selected[0]]=.17
            trained.gain[layer,:,selected[1]]=-.11
            trained.value_coefficients[layer].normal_(std=.025)
    reader,_,suffix,length,_=build_standalone(engine,layout,selected,teacher,trained)
    memory=reader.memory;memory.native_mode=False
    for parameter in memory.parameters():parameter.requires_grad_(True)
    source=teacher.document_cache
    stamps=[(tensor,tensor._version,tensor.clone()) for layer in source.layers
            for tensor in (layer.keys,layer.values) if tensor is not None]

    def start(compiled):
        return reader.compiled_branch(source) if compiled else fork_prefix(source,length)

    # Compare the entire question and two subsequent incremental positions.
    branches=[start(False),start(True)];outputs=[[],[]]
    with torch.no_grad():
        for index,branch in enumerate(branches):
            position=length
            for tokens in (suffix,torch.tensor([41],device=engine.device),
                           torch.tensor([42],device=engine.device)):
                hidden=private_forward(reader.reader,tokens,position,branch).last_hidden_state
                outputs[index].append(hidden);position+=len(tokens)
    for dynamic,materialized in zip(*outputs):
        torch.testing.assert_close(dynamic,materialized,
            atol=.03 if device=='cuda' else 2e-6,rtol=.05 if device=='cuda' else 2e-5)

    torch.manual_seed(313)
    probe=torch.randn_like(outputs[0][0])
    gradients=[]
    for compiled in (False,True):
        hidden=private_forward(reader.reader,suffix,length,start(compiled)).last_hidden_state
        gradients.append(torch.autograd.grad((hidden*probe).sum(),
            (memory.gain,memory.value_coefficients)))
    for dynamic,materialized in zip(*gradients):
        assert dynamic.isfinite().all() and dynamic.abs().sum()>0
        torch.testing.assert_close(dynamic,materialized,
            atol=.05 if device=='cuda' else 2e-6,rtol=.08 if device=='cuda' else 3e-5)
    for tensor,version,original in stamps:
        assert tensor._version==version
        torch.testing.assert_close(tensor,original,atol=0,rtol=0)
