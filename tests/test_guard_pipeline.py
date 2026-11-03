"""Verify a rejected adaptation returns the native answer with full timing."""
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
import masked_ttt.pipeline as pipeline
from test_batched_sd import ByteTokenizer


def test_rejected_candidate_returns_native_answer_and_waits_for_guard(monkeypatch):
    torch.manual_seed(35);torch.set_num_threads(1)
    model_config=Qwen3Config(vocab_size=512,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=1024,pad_token_id=0,eos_token_id=2)
    model_config._attn_implementation='eager'
    options=TTTConfig(chunk_tokens=16,max_context=1024,context_reserve_tokens=128,
        value_rank=2,steps=2,teacher_max_new_tokens=3,self_question_count=0,
        prefill_chunk_tokens=64,guard_enabled=True)
    engine=Backbone(Qwen3ForCausalLM(model_config).eval(),ByteTokenizer(),options)
    class Selector:
        def prepare(self,backbone,request):
            return PreparedRequest(request,prepare_layout(backbone,request),[1],dict(backend='unit_test'))
    monkeypatch.setattr(pipeline,'guard_answers',lambda *args:dict(use_adapted=False,
        reason='test_reject',reviews=[],seconds=0.,reference_answers_used=False))
    request=Request(document='The red folder is old. The blue folder is new. '*2,
                    question='Which folder is new?',max_new_tokens=3)
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result['prediction']==result['self_judge']['native_prediction']
    assert not result['self_judge']['use_adapted']
    assert result['timing']['ttft_seconds']>=result['timing']['candidate_first_token_seconds']
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['support']['LLM_self_judge_enabled']
