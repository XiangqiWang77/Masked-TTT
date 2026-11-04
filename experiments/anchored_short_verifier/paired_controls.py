"""Same-cache zero-update control for the original OLMo prompt format."""
from contextlib import contextmanager

import torch

from experiments.shared_question_sd.controls import _decode
from masked_ttt.standalone import build_standalone


@contextmanager
def original_question_controls():
    """Retain the production standalone materialization and release path."""
    import masked_ttt.pipeline as pipeline
    original=pipeline.answer_standalone
    original_decode=pipeline.decode_standalone

    @torch.no_grad()
    def control(engine,layout,selected,teacher,trained,limit):
        source=teacher.document_cache
        stamps=[(tensor,tensor._version,tensor.data_ptr()) for layer in source.layers
                for tensor in (layer.keys,layer.values) if tensor is not None]
        reader,branch,suffix,length,_=build_standalone(engine,layout,selected,teacher,trained)
        del branch
        if engine.config.standalone_answer_format!='original':
            raise ValueError('The matched comparison requires the original answer format')
        shared=bool(torch.equal(suffix,teacher.question_tokens))
        memory=reader.memory;adapted=_decode(engine,reader,source,suffix,length,limit)
        snapshot=memory.snapshot()
        try:
            memory.gain.zero_();memory.value_coefficients.zero_();memory.select_actual_read_path()
            zero=_decode(engine,reader,source,suffix,length,limit)
        finally:memory.restore(snapshot)
        if any(t._version!=version or t.data_ptr()!=pointer for t,version,pointer in stamps):
            raise AssertionError('The selected source cache changed during paired decoding')
        data=dict(zero=zero,adapted=adapted,same_selected_kv=True,same_question_suffix=True,
            teacher_suffix_matches_student=shared,same_generation_limit=True,
            teacher_answer_in_student_prompt=False,reference_answers_used=False)
        del reader,memory,source,stamps,snapshot,suffix
        ready=original(engine,layout,selected,teacher,trained,limit)
        if isinstance(ready,dict):
            if ready['generated_token_ids']!=adapted['tokens']:
                raise AssertionError('Standalone answer differs from the matched control')
            ready['original_question_controls']=data
        else:ready._original_question_controls=data
        return ready

    def decode(engine,ready):
        result=original_decode(engine,ready);data=ready._original_question_controls
        if result['generated_token_ids']!=data['adapted']['tokens']:
            raise AssertionError('Standalone answer differs from the matched control')
        result['original_question_controls']=data
        return result

    pipeline.answer_standalone=control;pipeline.decode_standalone=decode
    try:yield
    finally:
        pipeline.answer_standalone=original;pipeline.decode_standalone=original_decode
