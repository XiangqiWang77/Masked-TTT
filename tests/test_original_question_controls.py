from dataclasses import replace

import pytest

from experiments.anchored_short_verifier.paired_controls import original_question_controls
from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_original_prompt_controls_preserve_teacher_and_release_cache(architecture):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=0.,release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request=Request(document='The old note is red. The current note is blue. '*6,
                    question='Which color is current?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    method=MaskedTTT(engine,Selector());baseline=method.answer(request)
    with original_question_controls():result=method.answer(request)
    controls=result['standalone_cartridge']['original_question_controls']
    assert result['teacher']['tokens']==baseline['teacher']['tokens']
    assert result['generated_token_ids']==baseline['generated_token_ids']==controls['adapted']['tokens']
    assert controls['same_selected_kv'] and controls['same_question_suffix']
    assert controls['same_generation_limit'] and not controls['teacher_suffix_matches_student']
    assert not controls['teacher_answer_in_student_prompt'] and not controls['reference_answers_used']
    memory=result['standalone_cartridge']['memory']
    assert memory['additional_source_kv_bytes_before_decode']==0
    assert memory['additional_training_adapter_bytes_before_decode']==0
    assert result['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']==0
    assert result['original_document_KV_unchanged'] and result['backbone_unchanged']
    assert result['training']['objective']=='on-policy KL(teacher || student)'
    assert result['training']['selected_transfer_weight']==0


def test_controls_restore_hooks_when_an_error_occurs():
    import masked_ttt.pipeline as pipeline
    answer=pipeline.answer_standalone;decode=pipeline.decode_standalone
    with pytest.raises(RuntimeError,match='intentional'):
        with original_question_controls():raise RuntimeError('intentional')
    assert pipeline.answer_standalone is answer and pipeline.decode_standalone is decode
