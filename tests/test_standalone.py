"""The standalone cartridge contains selected KV plus learned read state."""
import torch
import pytest
from transformers import (Qwen3Config,Qwen3ForCausalLM,Olmo3Config,Olmo3ForCausalLM,
                          GptOssConfig,GptOssForCausalLM)

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt._layout import selected_read_plan
from masked_ttt.memory_accounting import storage_bytes
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from masked_ttt.standalone import selected_chunk_positions
from test_batched_sd import ByteTokenizer


def test_storage_counts_underlying_views_once():
    value=torch.zeros(10,10)
    assert storage_bytes([value,value[:2],value[3:]])==value.untyped_storage().nbytes()


def test_selected_slots_track_separators_and_original_order():
    ids=list(range(30));chunks=[[3,4,5],[6,7,8],[9,10,11],[12,13,14]]
    class Layout:
        initial=3
    layout=Layout();layout.ids=ids;layout.chunks=chunks
    plan=selected_read_plan(ids,chunks,3,[0,2],[40,41])
    assert selected_chunk_positions(layout,[0,2],plan,2)==[[3,4,5],[8,9,10]]
    assert plan.ids[3:11]==(3,4,5,40,41,9,10,11)


def make_engine():
    torch.manual_seed(47);torch.set_num_threads(1)
    config=Qwen3Config(vocab_size=512,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=2048,pad_token_id=0,eos_token_id=2)
    config._attn_implementation='eager'
    options=TTTConfig(chunk_tokens=32,max_context=2048,context_reserve_tokens=128,
        value_rank=2,steps=2,teacher_max_new_tokens=3,self_question_count=0,
        prefill_chunk_tokens=128,guard_enabled=False,standalone_cartridge_enabled=True)
    return Backbone(Qwen3ForCausalLM(config).eval(),ByteTokenizer(),options)


def test_standalone_decodes_with_selected_kv_and_reports_its_storage():
    engine=make_engine()
    document=('Filler information about another item. '*12+
              'The final answer is in the blue folder. '+
              'More unrelated information. '*12)
    request=Request(document=document,question='Where is the answer?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,original):
            layout=prepare_layout(backbone,original)
            return PreparedRequest(original,layout,[len(layout.chunks)//2],dict(backend='unit_test'))
    result=MaskedTTT(engine,Selector()).answer(request)
    standalone=result['standalone_cartridge'];sizes=standalone['memory']
    assert standalone['generated_token_ids']
    assert not sizes['raw_full_document_kv_used_at_decode']
    assert sizes['selected_kv_representation']=='independently_prefilled_selected_text'
    assert sizes['source_selected_kv_bytes']<result['memory']['full_document_kv_bytes']
    assert sizes['standalone_cartridge_bytes']==(sizes['materialized_selected_kv_bytes']+
                                                  sizes['cartridge_adapter_bytes'])
    assert result['memory']['full_kv_plus_adapter_bytes']>=result['memory']['full_document_kv_bytes']
    assert result['original_document_KV_unchanged'] and result['backbone_unchanged']


def test_selecting_all_chunks_gives_same_full_and_standalone_output():
    engine=make_engine()
    request=Request(document='A red note is old. A blue note is new. '*3,
                    question='Which note is new?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,original):
            layout=prepare_layout(backbone,original)
            return PreparedRequest(original,layout,list(range(len(layout.chunks))),dict(backend='unit_test'))
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result['standalone_cartridge']['prediction']==result['self_judge']['adapted_prediction']


@pytest.mark.parametrize('kind',['olmo3','gpt_oss'])
def test_standalone_runs_on_hybrid_backbones(kind):
    torch.manual_seed(71);torch.set_num_threads(1)
    common=dict(vocab_size=512,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
        layer_types=['sliding_attention','full_attention'],sliding_window=16,
        pad_token_id=0,eos_token_id=2)
    if kind=='olmo3':
        config=Olmo3Config(**common);model=Olmo3ForCausalLM
        tokenizer=ByteTokenizer()
    else:
        config=GptOssConfig(**common,head_dim=8,num_local_experts=2,num_experts_per_tok=1,
                            rope_scaling=None);model=GptOssForCausalLM
        class HarmonyTokenizer(ByteTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                return '<|start|>user<|message|>'+messages[0]['content']+'<|end|><|start|>assistant'
        tokenizer=HarmonyTokenizer()
    config._attn_implementation='eager'
    network=model(config).eval()
    if kind=='gpt_oss':
        for block in network.model.layers:
            with torch.no_grad():block.self_attn.sinks.fill_(0.2)
    options=TTTConfig(chunk_tokens=32,max_context=1024,context_reserve_tokens=128,
        value_rank=2,steps=2,teacher_max_new_tokens=3,self_question_count=0,
        prefill_chunk_tokens=64,guard_enabled=False,standalone_cartridge_enabled=True)
    engine=Backbone(network,tokenizer,options)
    request=Request(document='Old report says red. New report says blue. '*6,
                    question='What is the new report?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,original):
            layout=prepare_layout(backbone,original)
            return PreparedRequest(original,layout,[len(layout.chunks)//2],dict(backend='unit_test'))
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result['standalone_cartridge']['generated_token_ids']
    assert result['training']['adapted_layers']==[1]
    assert result['standalone_cartridge']['memory']['standalone_cartridge_bytes']>0
    assert result['original_document_KV_unchanged'] and result['backbone_unchanged']
