"""Compare BF16 and FP32 materialization of the same trained cartridge.

The primary result remains the ordinary BF16 standalone answer. The optional
float32 answer is decoded from an independent, separately materialized cache
after source and training states are released. This process-local diagnostic
must not be used concurrently with another request in the same process.
"""
from contextlib import contextmanager
from copy import copy
import gc

import torch

from experiments.precision_cartridge.materialization import (
    _float_values_during_materialization,_mixed_precision_attention,_paired_precision_controls,
)


def _rng_snapshot():
    return (torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:torch.cuda.set_rng_state_all(state[1])


def _release_ready(ready):
    if ready is not None:
        ready.reader=None;ready.branch=None;ready.suffix=None
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


@contextmanager
def capture_precision_paired():
    """Attach a same-trained-state precision diagnostic to the normal answer."""
    import masked_ttt.pipeline as pipeline
    from masked_ttt.standalone import StandaloneReady

    original_answer=pipeline.answer_standalone
    original_decode=pipeline.decode_standalone

    def prepare(engine,layout,selected,teacher,trained,limit):
        precision_ready=None;controls=None;error=None
        rng=_rng_snapshot()
        try:
            if trained.native_mode:
                error='no_active_update'
            elif engine.config.standalone_answer_format!='original':
                error='unsupported_answer_format'
            else:
                controls=_paired_precision_controls(engine,layout,selected,teacher,trained,limit)
                # A shallow Teacher view shares the independent selected source
                # only during materialization; no source KV is modified.
                teacher_view=copy(teacher)
                with _float_values_during_materialization():
                    precision_ready=original_answer(engine,layout,selected,teacher_view,trained,limit)
                del teacher_view
                if not isinstance(precision_ready,StandaloneReady):
                    raise RuntimeError('Deferred standalone decode is required')
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)
            _release_ready(precision_ready);precision_ready=None;controls=None
        finally:_restore_rng(rng)
        try:
            ready=original_answer(engine,layout,selected,teacher,trained,limit)
        except BaseException:
            _release_ready(precision_ready)
            raise
        if not isinstance(ready,StandaloneReady):
            _release_ready(precision_ready)
            raise RuntimeError('Deferred standalone decode is required')
        ready._precision_diagnostic=dict(ready=precision_ready,controls=controls,error=error)
        return ready

    def decode(engine,ready):
        diagnostic=ready._precision_diagnostic;ready._precision_diagnostic=None
        candidate=diagnostic['ready'];controls=diagnostic['controls']
        data=dict(passed=False,error=diagnostic['error'],reference_answers_used=False,
                  teacher_answer_in_student_prompt=False)
        rng=_rng_snapshot()
        try:
            if candidate is not None:
                dtypes=sorted({str(item.values.dtype) for item in candidate.branch.layers})
                with _mixed_precision_attention() as statistics:
                    result=original_decode(engine,candidate)
                memory=result['memory']
                if (result['generated_token_ids']!=controls['adapted']['tokens'] or
                    memory['additional_source_kv_bytes_before_decode']!=0 or
                    memory['additional_training_adapter_bytes_before_decode']!=0 or
                    'torch.float32' not in dtypes or statistics['float32_value_attention_calls']<1 or
                    not controls['same_selected_kv'] or not controls['same_question_suffix'] or
                    not controls['same_generation_limit'] or not controls['same_precision_backend']):
                    raise AssertionError('The precision cartridge or matched controls changed')
                data=dict(passed=True,error=None,precision_prediction=result['prediction'],
                    precision_tokens=result['generated_token_ids'],precision_zero=controls['zero'],
                    selected_source_bytes_before_decode=0,training_bytes_before_decode=0,
                    precision_cartridge_bytes=memory['standalone_cartridge_bytes'],
                    precision_value_dtypes=dtypes,
                    float_attention_calls=statistics['float32_value_attention_calls'],
                    same_selected_kv=True,same_question_suffix=True,same_generation_limit=True,
                    same_precision_backend=True,reference_answers_used=False,
                    teacher_answer_in_student_prompt=False,
                    paired_bf16_cartridge_resident_during_diagnostic=True)
        except Exception as exc:
            data=dict(passed=False,error=type(exc).__name__+': '+str(exc),
                      reference_answers_used=False,teacher_answer_in_student_prompt=False)
        finally:
            _restore_rng(rng);_release_ready(candidate)
        # Only the normal standalone artifact remains for the formal answer.
        result=original_decode(engine,ready)
        result['precision_paired_diagnostic']=data
        return result

    pipeline.answer_standalone=prepare
    pipeline.decode_standalone=decode
    try:yield
    finally:
        pipeline.answer_standalone=original_answer
        pipeline.decode_standalone=original_decode
