"""Check that answer positions after token eight can reach the SD objective."""
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from masked_ttt import Backbone,MaskedTTT,Request,TTTConfig
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_batched_sd import ByteTokenizer


def test_long_on_policy_rollout_scores_positions_after_eight():
    torch.manual_seed(53);torch.set_num_threads(1)
    cfg=Qwen3Config(vocab_size=512,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=1024,pad_token_id=0,eos_token_id=2)
    cfg._attn_implementation='eager'
    options=TTTConfig(chunk_tokens=16,max_context=1024,context_reserve_tokens=128,
        value_rank=2,steps=1,student_rollout_tokens=32,teacher_max_new_tokens=12,
        self_question_count=0,prefill_chunk_tokens=64,guard_enabled=False)
    engine=Backbone(Qwen3ForCausalLM(cfg).eval(),ByteTokenizer(),options)
    engine.eos_ids=set()
    class Selector:
        def prepare(self,backbone,request):
            return PreparedRequest(request,prepare_layout(backbone,request),[1],dict(backend='unit_test'))
    request=Request(document='The red folder is old. The blue folder is new. '*2,
                    question='Which folder is new?',max_new_tokens=12)
    result=MaskedTTT(engine,Selector()).answer(request)
    step=result['training']['steps'][0]
    assert len(step['student_rollout_tokens'])==12
    assert step['target_positions']==list(range(12)) and 8 in step['target_positions']
    assert result['training']['teacher_conditioned_on_student_prefix']
    assert result['training']['student_rollout_limit']==32
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
