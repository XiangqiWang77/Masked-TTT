"""Emit a deterministic tiny GPT-OSS zero-transfer trace for archive comparison."""
import json

import torch
from transformers import GptOssConfig,GptOssForCausalLM

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_batched_sd import ByteTokenizer


def main():
    torch.manual_seed(71);torch.set_num_threads(1)
    config=GptOssConfig(vocab_size=512,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
        layer_types=["sliding_attention","full_attention"],sliding_window=16,pad_token_id=0,
        eos_token_id=2,head_dim=8,num_local_experts=2,num_experts_per_tok=1,rope_scaling=None)
    config._attn_implementation="eager"
    model=GptOssForCausalLM(config).eval()
    for block in model.model.layers:
        with torch.no_grad():block.self_attn.sinks.fill_(.2)
    class HarmonyTokenizer(ByteTokenizer):
        def apply_chat_template(self,messages,**kwargs):
            return "<|start|>user<|message|>"+messages[0]["content"]+"<|end|><|start|>assistant"
    options=TTTConfig(chunk_tokens=32,max_context=1024,context_reserve_tokens=128,value_rank=2,
        steps=2,teacher_max_new_tokens=3,self_question_count=0,prefill_chunk_tokens=64,
        guard_enabled=False,standalone_cartridge_enabled=True)
    engine=Backbone(model,HarmonyTokenizer(),options)
    request=Request(document="Old report says red. New report says blue. "*6,
                    question="What is the new report?",max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,original):
            layout=prepare_layout(backbone,original)
            return PreparedRequest(original,layout,[1,len(layout.chunks)-2],dict(backend="unit_test"))
    result=MaskedTTT(engine,Selector()).answer(request)
    print(json.dumps(dict(prediction=result["prediction"],
        standalone=result["standalone_cartridge"]["prediction"],
        teacher=result["teacher"]["prediction"],steps=result["training"]["steps"],
        frozen=result["backbone_unchanged"],cache_unchanged=result["original_document_KV_unchanged"]),
        sort_keys=True))


if __name__=="__main__":main()
