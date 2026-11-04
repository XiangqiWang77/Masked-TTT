import copy

import pytest

from experiments.kv_only_cartridge.freeze_real_evidence import RELEASE_CHECKS,REQUEST_CHECKS,freeze


def fixture():
    def release(model,gpu):
        return dict(model=model,gpu=gpu,passed=True,long_attention_backend='flash',
            long_attention_positions=32,evaluated_positions=3,adapter_bytes_after=0,
            online_api_calls=0,reference_answers_used=False,**{key:True for key in RELEASE_CHECKS})
    qwen=release('Qwen/Qwen3-4B-Instruct-2507','NVIDIA H200')
    gpt=release('openai/gpt-oss-20b','NVIDIA B200')
    request=dict(model='qwen',model_type='qwen3',identity={'gpu':'NVIDIA B200'},passed=True,
        generic_teacher_answer_in_prompt=False,online_api_calls=0,reference_answers_used=False,
        **{key:True for key in REQUEST_CHECKS})
    return qwen,gpt,request


def test_freeze_distinguishes_real_release_from_pending_request_local_gpt():
    data=freeze(*fixture())
    assert data['kv_only_release']['qwen']['passed'] and data['kv_only_release']['gpt']['passed']
    assert data['request_local']['qwen']['training_gradients_match']
    assert data['request_local']['gpt']['state']=='pending_real_gpu_check'
    assert data['new_model_calls']==0 and not data['reference_answers_used']


def test_freeze_rejects_missing_or_non_gpu_evidence():
    a,b,c=copy.deepcopy(fixture());b['same_standalone_logits']=False
    with pytest.raises(ValueError):freeze(a,b,c)
    a,b,c=copy.deepcopy(fixture());b['gpu']='cpu'
    with pytest.raises(ValueError):freeze(a,b,c)
    a,b,c=copy.deepcopy(fixture());c['same_generic_tokens_before_after_release']=False
    with pytest.raises(ValueError):freeze(a,b,c)
