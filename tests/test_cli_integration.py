"""Exercise the packaged CLI with real local model loading and frozen scores."""
import json
import sys

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

import masked_ttt.cli as cli
from masked_ttt.decision_api import DecisionClient


def test_local_cli_loads_model_and_cached_scores_without_network(tmp_path,monkeypatch):
    torch.manual_seed(23);torch.set_num_threads(1)
    vocabulary=['[UNK]','[PAD]','[EOS]']+sorted(ByteLevel.alphabet())
    tokenizer=Tokenizer(BPE({word:i for i,word in enumerate(vocabulary)},[],unk_token='[UNK]'))
    tokenizer.pre_tokenizer=ByteLevel(add_prefix_space=False)
    tokenizer.decoder=ByteLevelDecoder()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',
                                 eos_token='[EOS]')
    fast.chat_template=('{% for message in messages %}user {{ message.content }}\n{% endfor %}'
                        '{% if add_generation_prompt %}assistant{% endif %}')
    model_path=tmp_path/'model';model_path.mkdir()
    fast.save_pretrained(model_path)
    config=Qwen3Config(vocab_size=len(fast),hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=512,pad_token_id=fast.pad_token_id,eos_token_id=fast.eos_token_id)
    Qwen3ForCausalLM(config).save_pretrained(model_path)

    first='The red box is empty. ';second='The blue box is full.'
    document=first+second;question='Where is the blue box?'
    request=dict(document=document,question=question,
        prompt='Context: '+document+'\nQuestion: '+question,max_new_tokens=3)
    request_path=tmp_path/'request.json';request_path.write_text(json.dumps(request))
    scores=tmp_path/'cached';(scores/'scores').mkdir(parents=True)
    (scores/'case_to_score_file.json').write_text(json.dumps({'case-1':'case.json'}))
    scored=dict(schema='masked-ttt-source-scores-v1',complete=True,case_key='case-1',
        question=question,reference_answers_used=False,source_spans=[[0,len(first)],[len(first),len(document)]],
        records=[dict(chunk=0,utility_output_valid=True,decision_probabilities=dict(yes=.1,no=.9)),
                 dict(chunk=1,utility_output_valid=True,decision_probabilities=dict(yes=.9,no=.1))],
        canonical_tokenizer='synthetic',scoring_seconds=0,estimated_cost_usd=0)
    (scores/'scores/case.json').write_text(json.dumps(scored))
    config_path=tmp_path/'settings.json'
    config_path.write_text(json.dumps(dict(chunk_tokens=16,keep_fraction=.5,max_context=512,
        context_reserve_tokens=32,prefill_chunk_tokens=32,value_rank=2,steps=2,max_answer_steps=2,
        teacher_max_new_tokens=3,student_rollout_tokens=3,self_question_count=0,
        guard_enabled=False,matched_cache_direct=True,matched_cache_final_suffix='practice_gated_format')))
    output=tmp_path/'output.json'
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    def forbid_network(*args,**kwargs):raise AssertionError('Cached-score runs must not call Decision API')
    monkeypatch.setattr(DecisionClient,'__init__',forbid_network)
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model',str(model_path),'--request',str(request_path),
        '--score-json-dir',str(scores),'--case-id','case-1','--config-json',str(config_path),
        '--device','cpu','--local-files-only','--output',str(output)])
    cli.main()
    result=json.loads(output.read_text())
    assert result['selection']['selected_chunks']==[1]
    assert result['selection']['online_api_calls']==0 and not result['selection']['reference_answers_used']
    assert result['training']['training_cache']=='selected_document'
    assert result['training']['accepted_steps']==2
    assert all(step['gain_gradient_norm']>0 and step['value_gradient_norm']>0
               for step in result['training']['steps'])
    assert result['standalone_cartridge']['selected_output_arm']=='zero_format'
    assert result['prediction']==result['matched_cache_study']['zero_format']['prediction']
    assert not result['standalone_cartridge']['teacher_answer_in_final_prompt']
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['matched_cache_study']['source_selected_cache_unchanged']
