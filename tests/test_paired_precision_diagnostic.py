from dataclasses import replace

import pytest
import torch

from experiments.precision_cartridge.paired_diagnostic import capture_precision_paired
from experiments.shared_question_sd.alignment import shared_config,shared_request,verify_shared_training_suffix
from experiments.shared_question_sd.controls import aligned_controls
from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def make_request(engine):
    request=Request(document='Old note is red. Current note is blue. '*6,
                    question='Which color is current?',max_new_tokens=3)
    instruction=engine.config.teacher_response_instruction
    engine.config=shared_config(engine.config)
    return shared_request(request,instruction)


class Selector:
    def prepare(self,backbone,item):
        layout=prepare_layout(backbone,item)
        return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_matched_precision_diagnostic_preserves_training_and_normal_answer(architecture):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.model.to(dtype=torch.bfloat16)
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=0.,release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request=make_request(engine);method=MaskedTTT(engine,Selector())
    torch.manual_seed(119)
    with aligned_controls(),verify_shared_training_suffix():baseline=method.answer(request)
    after_baseline=torch.get_rng_state().clone()
    torch.manual_seed(119)
    with capture_precision_paired(),aligned_controls(),verify_shared_training_suffix():
        result=method.answer(request)
    assert torch.equal(after_baseline,torch.get_rng_state())
    assert result['teacher']['tokens']==baseline['teacher']['tokens']
    assert result['generated_token_ids']==baseline['generated_token_ids']
    assert result['standalone_cartridge']['aligned_controls']==baseline['standalone_cartridge']['aligned_controls']
    for actual,expected in zip(result['training']['steps'],baseline['training']['steps']):
        assert actual['student_rollout_tokens']==expected['student_rollout_tokens']
        for key in ('teacher_KL_before','gain_gradient_norm','value_gradient_norm'):
            assert actual[key]==pytest.approx(expected[key],abs=1e-8)
    info=result['standalone_cartridge']['precision_paired_diagnostic']
    assert info['passed'] and info['same_selected_kv'] and info['same_question_suffix']
    assert info['same_generation_limit'] and info['same_precision_backend']
    assert info['selected_source_bytes_before_decode']==info['training_bytes_before_decode']==0
    assert info['paired_bf16_cartridge_resident_during_diagnostic']
    assert not info['reference_answers_used'] and not info['teacher_answer_in_student_prompt']
    assert result['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']==0
    normal=result['standalone_cartridge']['memory']
    assert normal['additional_source_kv_bytes_before_decode']==0
    assert normal['additional_training_adapter_bytes_before_decode']==0


def test_optional_precision_failure_does_not_replace_formal_answer(monkeypatch):
    import experiments.precision_cartridge.paired_diagnostic as paired
    engine=make_engine();engine.model.to(dtype=torch.bfloat16)
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=0.,release_full_before_cartridge=True)
    request=make_request(engine);method=MaskedTTT(engine,Selector())
    with aligned_controls(),verify_shared_training_suffix():baseline=method.answer(request)
    def fail(*args,**kwargs):raise RuntimeError('synthetic diagnostic failure')
    monkeypatch.setattr(paired,'_paired_precision_controls',fail)
    with capture_precision_paired(),aligned_controls(),verify_shared_training_suffix():
        result=method.answer(request)
    assert result['generated_token_ids']==baseline['generated_token_ids']
    info=result['standalone_cartridge']['precision_paired_diagnostic']
    assert not info['passed'] and 'synthetic diagnostic failure' in info['error']


def test_hooks_restore_on_failure():
    import masked_ttt.pipeline as pipeline
    answer=pipeline.answer_standalone;decode=pipeline.decode_standalone
    with pytest.raises(RuntimeError,match='intentional'):
        with capture_precision_paired():raise RuntimeError('intentional')
    assert pipeline.answer_standalone is answer and pipeline.decode_standalone is decode
