"""Check selected-cache disposal on the real CUDA attention path."""
from dataclasses import replace
import math

import pytest
import torch

from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires a CUDA GPU')
@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_release_selected_source_cuda_preserves_answer_and_memory(architecture):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.model.to(device='cuda:0',dtype=torch.bfloat16)
    engine.device=torch.device('cuda:0')
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        release_full_before_cartridge=True,release_selected_source_before_decode=False)
    request=Request(document='Old note says red. Current note says blue. '*6,
                    question='What is in the current note?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    method=MaskedTTT(engine,Selector())
    retained=method.answer(request)
    engine.config=replace(engine.config,release_selected_source_before_decode=True)
    released=method.answer(request)
    before=retained['standalone_cartridge']['memory']
    after=released['standalone_cartridge']['memory']
    assert released['generated_token_ids']==retained['generated_token_ids']
    assert len(released['training']['steps'])==len(retained['training']['steps'])
    for a,b in zip(released['training']['steps'],retained['training']['steps']):
        assert a['student_rollout_tokens']==b['student_rollout_tokens']
        for key in ('teacher_KL_before','gain_gradient_norm','value_gradient_norm'):
            assert math.isclose(a[key],b[key],rel_tol=1e-3,abs_tol=1e-4)
    assert before['additional_source_kv_bytes_before_decode']>0
    assert after['additional_source_kv_bytes_before_decode']==0
    assert after['additional_training_adapter_bytes_before_decode']==0
    assert after['live_document_state_before_decode_bytes']<before['live_document_state_before_decode_bytes']
    assert after['cuda_allocated_before_decode_bytes']<=after['cuda_allocated_before_source_release_bytes']
    assert after['cuda_reserved_before_decode_bytes']<=after['cuda_reserved_before_source_release_bytes']
