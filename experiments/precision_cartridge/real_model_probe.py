"""Read-only paired smoke check on an already-loaded backbone."""
from dataclasses import replace
import math

from experiments.precision_cartridge.materialization import float32_cartridge
from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest


def validate_loaded_backbone(engine):
    """Compare training and selected-only decoding with one synthetic request.

    This verifies execution and memory release. It is not a benchmark or an
    accuracy claim. No external data, answer labels, or online API are used.
    """
    request=Request(document='The old note is red. '+'Ordinary material. '*30+
                    'The current note is blue.',
                    question='What color is the current note?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[len(layout.chunks)//2],dict(backend='synthetic'))
    old=engine.config
    try:
        engine.config=replace(old,chunk_tokens=64,steps=2,max_answer_steps=2,
            self_question_count=0,student_rollout_tokens=3,guard_enabled=False,
            selected_transfer_weight=0.,value_basis_source='document',
            release_full_before_cartridge=True,release_selected_source_before_decode=True)
        method=MaskedTTT(engine,Selector())
        baseline=method.answer(request,time_budget_seconds=300)
        with float32_cartridge(matched_controls=True):precise=method.answer(request,time_budget_seconds=300)
        before=baseline['training']['steps'];after=precise['training']['steps']
        same=(len(before)==len(after) and baseline['teacher']['tokens']==precise['teacher']['tokens'] and
              baseline['full_context']['generated_token_ids']==precise['full_context']['generated_token_ids'])
        for first,second in zip(before,after):
            same &= first['student_rollout_tokens']==second['student_rollout_tokens']
            for key in ('teacher_KL_before','gain_gradient_norm','value_gradient_norm'):
                same &= math.isclose(first[key],second[key],rel_tol=1e-5,abs_tol=1e-7)
        memory=precise['standalone_cartridge']['memory']
        matched=precise['standalone_cartridge']['precision_matched_controls']
        if (not same or not before or
            not baseline['backbone_unchanged'] or not precise['backbone_unchanged'] or
            not baseline['original_document_KV_unchanged'] or not precise['original_document_KV_unchanged'] or
            precise['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']!=0 or
            memory['additional_source_kv_bytes_before_decode']!=0 or
            memory['additional_training_adapter_bytes_before_decode']!=0 or
            memory['experimental_materialization']!='float32_value_cache' or
            memory['experimental_float_attention_calls']<1 or
            not matched['same_selected_kv'] or not matched['same_question_suffix'] or
            not matched['same_generation_limit'] or not matched['same_precision_backend'] or
            matched['reference_answers_used'] or matched['teacher_answer_in_student_prompt']):
            raise AssertionError('Precision materialization did not preserve training and release contracts')
        return dict(passed=True,same_teacher_and_training=True,
            same_full_prediction=True,full_and_source_state_released=True,
            baseline_tokens=baseline['standalone_cartridge']['generated_token_ids'],
            precision_tokens=precise['standalone_cartridge']['generated_token_ids'],
            matched_zero_tokens=matched['zero']['tokens'],
            baseline_cartridge_bytes=baseline['standalone_cartridge']['memory']['standalone_cartridge_bytes'],
            precision_cartridge_bytes=memory['standalone_cartridge_bytes'],
            value_dtypes=memory['experimental_value_dtypes'],
            float_attention_calls=memory['experimental_float_attention_calls'],
            reference_answers_used=False,online_api_calls=0,
            note='Synthetic smoke test; no answer-quality comparison')
    finally:engine.config=old
