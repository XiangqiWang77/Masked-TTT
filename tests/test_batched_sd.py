"""Causal teacher-forcing parity for full and hybrid attention backbones."""
from types import SimpleNamespace

import pytest
import torch
from transformers import (Qwen3Config,Qwen3ForCausalLM,Olmo3Config,Olmo3ForCausalLM,
                          GptOssConfig,GptOssForCausalLM)

from masked_ttt import Backbone,TTTConfig,MaskedTTT,Request
from masked_ttt.batched_sd import batched_teacher_logits
from masked_ttt.cache import AppendOnlyDocumentCache,fork_prefix
from masked_ttt.checkpointing import layer_checkpointed_logits
from masked_ttt.memory import KVReadMemory
from masked_ttt.reader import MemoryReader,matched_incremental_logits
from masked_ttt.runtime import private_forward
from masked_ttt.teacher import TeacherSignal,score_student_prefix
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest


@pytest.fixture(params=['qwen3','olmo3','gpt_oss'])
def setup(request):
    torch.manual_seed(107);torch.set_num_threads(1)
    common=dict(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=256,
                pad_token_id=0,eos_token_id=2)
    if request.param=='qwen3':
        config=Qwen3Config(**common,head_dim=8);cls=Qwen3ForCausalLM
    elif request.param=='olmo3':
        config=Olmo3Config(**common,layer_types=['sliding_attention','full_attention'],sliding_window=4)
        cls=Olmo3ForCausalLM
    else:
        config=GptOssConfig(**common,head_dim=8,num_local_experts=2,num_experts_per_tok=1,
                            rope_scaling=None,layer_types=['sliding_attention','full_attention'],sliding_window=4)
        cls=GptOssForCausalLM
    config._attn_implementation='eager';model=cls(config).eval()
    if request.param=='gpt_oss':
        for block in model.model.layers:
            with torch.no_grad():block.self_attn.sinks.fill_(.2)
    engine=Backbone(model,SimpleNamespace(eos_token_id=2),
                    TTTConfig(max_context=256,context_reserve_tokens=32,
                              prefill_chunk_tokens=4,value_rank=2))
    document=torch.tensor([5,6,7,8,9,10,11,12]);cache=AppendOnlyDocumentCache(config,len(document))
    with torch.no_grad():
        private_forward(engine.model,document[:4],0,cache)
        private_forward(engine.model,document[4:],4,cache)
    memory=KVReadMemory(config,[list(range(4)),list(range(4,8))],[1],len(document),rank=2,sink_tokens=1)
    memory.prepare_value_basis(cache,memory.adapted_layers);memory.value_enabled=True;memory.native_mode=False
    with torch.no_grad():
        for layer in memory.adapted_layers:
            memory.gain[layer,:,1].fill_(.15)
            memory.value_coefficients[layer].fill_(.03)
    return engine,cache,memory,MemoryReader(engine,memory)


def test_batched_matches_incremental_and_does_not_see_future(setup):
    engine,cache,memory,reader=setup
    suffix=torch.tensor([20,21,22]);targets=torch.tensor([30,31,32])
    before=[(x,x.clone(),x._version) for layer in cache.layers for x in (layer.keys,layer.values)]
    with torch.no_grad():
        reference=matched_incremental_logits(engine,reader,cache,suffix,targets)
        batched=batched_teacher_logits(engine,reader,cache,suffix,targets)
        changed=batched_teacher_logits(engine,reader,cache,suffix,torch.tensor([30,55,56]))
    torch.testing.assert_close(batched,reference,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(changed[:2],batched[:2],atol=1e-7,rtol=1e-6)
    for tensor,old,version in before:
        assert tensor._version==version
        torch.testing.assert_close(tensor,old,atol=0,rtol=0)


def test_batched_joint_gradients_match_incremental(setup):
    engine,cache,memory,reader=setup
    suffix=torch.tensor([20,21,22]);targets=torch.tensor([30,31,32])
    params=(memory.gain,memory.value_coefficients)
    logits=layer_checkpointed_logits(engine,reader,cache,suffix,targets)
    grads=torch.autograd.grad(logits.square().mean(),params)
    batched=batched_teacher_logits(engine,reader,cache,suffix,targets)
    new_grads=torch.autograd.grad(batched.square().mean(),params)
    for expected,actual in zip(grads,new_grads):
        assert expected.float().norm()>0
        torch.testing.assert_close(actual,expected,atol=3e-6,rtol=5e-4)
    assert not bool(new_grads[0][...,:1].any())
    assert all(not p.requires_grad and p.grad is None for p in engine.model.parameters())


def test_answer_position_uses_the_same_causal_logits_and_gradients(setup):
    engine,cache,memory,reader=setup
    suffix=torch.tensor([20,21,22]);targets=torch.tensor([30,31,32]);index=torch.tensor([1])
    params=(memory.gain,memory.value_coefficients)
    reference=layer_checkpointed_logits(engine,reader,cache,suffix,targets,target_positions=index)
    expected=torch.autograd.grad(reference.square().mean(),params)
    focused=batched_teacher_logits(engine,reader,cache,suffix,targets,target_positions=index)
    actual=torch.autograd.grad(focused.square().mean(),params)
    torch.testing.assert_close(focused,reference,atol=3e-6,rtol=3e-5)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=3e-6,rtol=5e-4)


def test_teacher_scores_student_prefix_causally_and_matches_incremental(setup):
    engine,cache,memory,reader=setup
    native=SimpleNamespace(model=engine.model.model,lm_head=engine.model.lm_head)
    suffix=torch.tensor([20,21,22]);trajectory=torch.tensor([30,31,32])
    teacher=TeacherSignal(trajectory,torch.empty(0),'',False,0.,0.,0,False,
                          cache,suffix,memory.document_length)
    with torch.no_grad():
        scored=score_student_prefix(engine,native,teacher,trajectory)
        altered=score_student_prefix(engine,native,teacher,torch.tensor([30,55,56]))
        branch=fork_prefix(cache,memory.document_length)
        output=private_forward(native,suffix,memory.document_length,branch)
        values=[]
        for index in range(len(trajectory)):
            values.append(engine.model.lm_head(output.last_hidden_state[0,-1:]).float().softmax(-1))
            if index+1<len(trajectory):
                output=private_forward(native,trajectory[index:index+1],
                    memory.document_length+len(suffix)+index,branch)
    torch.testing.assert_close(scored,torch.cat(values),atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(scored[:2],altered[:2],atol=1e-7,rtol=1e-6)


class ByteTokenizer:
    eos_token_id=2
    def apply_chat_template(self,messages,**kwargs):
        return 'USER:\n'+messages[0]['content']+'\nASSISTANT:\n'
    def encode(self,text,**kwargs):return list(text.encode('ascii'))
    def decode(self,tokens,**kwargs):
        return ''.join(chr(i) if 32<=i<127 else f'<{i}>' for i in tokens)
    def __call__(self,text,**kwargs):
        return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])


def test_simple_selected_sd_keeps_backbone_and_recomputes_question(monkeypatch):
    import json
    import masked_ttt.pipeline as pipeline
    torch.manual_seed(14);torch.set_num_threads(1)
    config=Qwen3Config(vocab_size=512,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=1024,
        pad_token_id=0,eos_token_id=2)
    config._attn_implementation='eager'
    options=TTTConfig(chunk_tokens=16,max_context=1024,context_reserve_tokens=128,value_rank=2,
                      steps=2,teacher_max_new_tokens=3,prefill_chunk_tokens=64,
                      self_question_count=0,guard_enabled=False,
                      teacher_response_instruction='Return only the final answer.')
    engine=Backbone(Qwen3ForCausalLM(config).eval(),ByteTokenizer(),options)
    class Selector:
        def prepare(self,backbone,request):
            return PreparedRequest(request,prepare_layout(backbone,request),[1,3],
                                   dict(complete_original_document=True))
    request=Request(document='X drafted 300. X finally received 320. '*3,
                    question='Which budget was finally approved?',max_new_tokens=3)
    calls=[];forward=pipeline.private_forward
    def observe(reader,tokens,start,branch):
        calls.append((tokens.clone(),start));return forward(reader,tokens,start,branch)
    monkeypatch.setattr(pipeline,'private_forward',observe)
    rollout_calls=[];real_rollout=pipeline.student_rollout
    def watch_rollout(*args,**kwargs):
        tokens=real_rollout(*args,**kwargs);rollout_calls.append(tokens.clone());return tokens
    monkeypatch.setattr(pipeline,'student_rollout',watch_rollout)
    scored=[];real_score=pipeline.score_student_prefix
    def watch_score(engine,reader,teacher,tokens):
        scored.append(tokens.clone());return real_score(engine,reader,teacher,tokens)
    monkeypatch.setattr(pipeline,'score_student_prefix',watch_score)
    before=[p.detach().clone() for p in engine.model.parameters()]
    method=MaskedTTT(engine,Selector());result=method.answer(request)
    json.dumps(result,allow_nan=False)
    assert result['training']['accepted_steps']==2
    assert all(step['gain_gradient_norm']>0 and step['value_gradient_norm']>0
               for step in result['training']['steps'])
    assert result['training']['causal_batched_teacher_forcing']
    assert result['training']['student_rollouts_refreshed_each_step']
    assert result['training']['teacher_conditioned_on_student_prefix']
    assert len(rollout_calls)==len(scored)==len(result['training']['steps'])==2
    for rollout,teacher_tokens,step in zip(rollout_calls,scored,result['training']['steps']):
        torch.testing.assert_close(rollout,teacher_tokens,atol=0,rtol=0)
        assert step['student_rollout_tokens']==rollout.tolist()
    assert not result['support']['LLM_self_judge_enabled']
    assert result['teacher']['fresh_selected_context'] and result['teacher']['target_fixed']
    assert result['teacher']['on_policy_targets_recomputed']
    assert 'student prefix' in result['teacher']['target_fixed_scope']
    assert result['teacher']['first_token_scope']=='selected_prompt_planning_prefill_first_token'
    assert (result['teacher']['first_token_seconds'] <=
            result['timing']['icl_teacher_first_token_seconds_segmented'] <=
            result['timing']['ttft_seconds'])
    assert result['teacher']['response_instruction_applied']
    assert result['teacher']['prompt_tokens']<len(prepare_layout(engine,request).ids)
    assert result['final_question_recomputed'] and not result['teacher_answer_in_final_prompt']
    layout=prepare_layout(engine,request)
    torch.testing.assert_close(calls[0][0],torch.tensor(layout.ids[layout.document_end:]),atol=0,rtol=0)
    assert calls[0][1]==layout.document_end
    for parameter,previous in zip(engine.model.parameters(),before):
        torch.testing.assert_close(parameter,previous,atol=0,rtol=0)
        assert not parameter.requires_grad and parameter.grad is None
    assert not hasattr(method,'memory')
