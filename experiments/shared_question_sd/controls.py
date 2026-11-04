"""Matched zero-update control while retaining true standalone final decoding."""
from contextlib import contextmanager

import torch

from masked_ttt.runtime import private_forward
from masked_ttt.standalone import build_standalone


@torch.no_grad()
def _decode(engine,reader,source,suffix,length,limit):
    reader.memory.select_actual_read_path();branch=reader.compiled_branch(source)
    engine.budget.check();output=private_forward(reader.reader,suffix,length,branch)
    tokens=[]
    while len(tokens)<limit:
        token=int(engine.model.lm_head(output.last_hidden_state[0,-1:])[0].argmax())
        tokens.append(token)
        if token in engine.eos_ids or len(tokens)==limit:break
        engine.budget.check()
        output=private_forward(reader.reader,torch.tensor([token],device=engine.device),
                               length+len(suffix)+len(tokens)-1,branch)
    return dict(prediction=engine.tokenizer.decode(tokens,skip_special_tokens=True),tokens=tokens)


@contextmanager
def aligned_controls():
    """Keep the production materialization/release path for the final answer."""
    import masked_ttt.pipeline as pipeline
    original=pipeline.answer_standalone
    original_decode=getattr(pipeline,'decode_standalone',None)

    @torch.no_grad()
    def control(engine,layout,selected,teacher,trained,limit):
        source=teacher.document_cache
        stamps=[(x,x._version,x.data_ptr()) for layer in source.layers
                for x in (layer.keys,layer.values) if x is not None]
        reader,branch,suffix,length,_=build_standalone(engine,layout,selected,teacher,trained)
        del branch
        if not torch.equal(suffix,teacher.question_tokens):
            raise AssertionError('Teacher and final Student question suffixes differ')
        memory=reader.memory
        adapted=_decode(engine,reader,source,suffix,length,limit)
        snapshot=memory.snapshot()
        try:
            memory.gain.zero_();memory.value_coefficients.zero_();memory.select_actual_read_path()
            zero=_decode(engine,reader,source,suffix,length,limit)
        finally:memory.restore(snapshot)
        if any(t._version!=version or t.data_ptr()!=pointer for t,version,pointer in stamps):
            raise AssertionError('The selected source cache changed during control decoding')
        data=dict(zero=zero,adapted=adapted,shared_question_suffix=True,
            same_selected_kv=True,same_generation_limit=True,
            teacher_answer_in_student_prompt=False,reference_answers_used=False)
        # Release diagnostic references before the real materialization. The
        # new runtime then drops both source and training state before decoding.
        del reader,memory,source,stamps,snapshot,suffix
        result=original(engine,layout,selected,teacher,trained,limit)
        if isinstance(result,dict):
            if result['generated_token_ids']!=adapted['tokens']:
                raise AssertionError('Standalone answer differs from matched control')
            result['aligned_controls']=data
        else:
            result._aligned_controls=data
        return result

    def decode(engine,ready):
        result=original_decode(engine,ready);data=ready._aligned_controls
        if result['generated_token_ids']!=data['adapted']['tokens']:
            raise AssertionError('Standalone answer differs from matched control')
        result['aligned_controls']=data
        return result

    pipeline.answer_standalone=control
    if original_decode is not None:pipeline.decode_standalone=decode
    try:yield
    finally:
        pipeline.answer_standalone=original
        if original_decode is not None:pipeline.decode_standalone=original_decode
