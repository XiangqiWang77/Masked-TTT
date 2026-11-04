"""Preserve a trained value correction in a float32 standalone cache.

This is an experimental, single-request context. It leaves the SD training
path and both source caches unchanged. The standalone answer may use a
slower attention implementation and more cache memory, so effectiveness is
measured independently of the normal BF16 materialization.
"""
from contextlib import contextmanager

import torch


@contextmanager
def _float_values_during_materialization():
    from masked_ttt.reader import MemoryReader

    original=MemoryReader.compiled_branch

    def compile_float_values(reader,cache):
        memory=reader.memory
        had_instance_override='values' in memory.__dict__
        previous_override=memory.__dict__.get('values')
        normal_values=memory.values
        # Calculate V + delta in float32 rather than rounding the correction
        # into the source cache's BF16 representation before the addition.
        memory.values=lambda layer,native:normal_values(layer,native.float())
        try:return original(reader,cache)
        finally:
            if had_instance_override:memory.values=previous_override
            else:del memory.values

    MemoryReader.compiled_branch=compile_float_values
    try:yield
    finally:MemoryReader.compiled_branch=original


@contextmanager
def _mixed_precision_attention():
    import masked_ttt.reader as reader_module
    import masked_ttt.architectures as architecture_module

    reader_attention=reader_module.native_attention
    hybrid_attention=architecture_module.native_attention
    statistics=dict(float32_value_attention_calls=0)

    def wrap(original):
        def attention(q,k,v,**options):
            if v.dtype==torch.float32 and q.dtype in (torch.float16,torch.bfloat16):
                # FlashAttention does not accept float32 V with BF16 Q/K. The
                # eager path implements the same causal/GQA/sink computation.
                opts=dict(options,backend='eager')
                statistics['float32_value_attention_calls']+=1
                return original(q.float(),k.float(),v,**opts).to(q.dtype)
            return original(q,k,v,**options)
        return attention

    reader_module.native_attention=wrap(reader_attention)
    architecture_module.native_attention=wrap(hybrid_attention)
    try:yield statistics
    finally:
        reader_module.native_attention=reader_attention
        architecture_module.native_attention=hybrid_attention


@torch.no_grad()
def _paired_precision_controls(engine,layout,selected,teacher,trained,limit):
    from masked_ttt.runtime import private_forward
    from masked_ttt.standalone import build_standalone

    source=teacher.document_cache
    stamps=[(tensor,tensor._version,tensor.data_ptr()) for layer in source.layers
            for tensor in (layer.keys,layer.values) if tensor is not None]
    with _float_values_during_materialization(),_mixed_precision_attention():
        reader,branch,suffix,length,_=build_standalone(engine,layout,selected,teacher,trained)
        del branch
        memory=reader.memory

        def answer():
            # Force exactly the same augmented float32-value path even when
            # the comparison state is zero, so only the learned state changes.
            memory.native_mode=False
            local=reader.compiled_branch(source)
            engine.budget.check();output=private_forward(reader.reader,suffix,length,local)
            tokens=[]
            while len(tokens)<limit:
                token=int(engine.model.lm_head(output.last_hidden_state[0,-1:])[0].argmax())
                tokens.append(token)
                if token in engine.eos_ids or len(tokens)==limit:break
                engine.budget.check()
                output=private_forward(reader.reader,torch.tensor([token],device=engine.device),
                    length+len(suffix)+len(tokens)-1,local)
            return dict(tokens=tokens,prediction=engine.tokenizer.decode(tokens,skip_special_tokens=True))

        snapshot=memory.snapshot()
        try:
            adapted=answer()
            memory.gain.zero_();memory.value_coefficients.zero_()
            zero=answer()
        finally:memory.restore(snapshot)
        if any(t._version!=version or t.data_ptr()!=pointer for t,version,pointer in stamps):
            raise AssertionError('The independent selected source was modified')
    return dict(zero=zero,adapted=adapted,same_selected_kv=True,same_question_suffix=True,
                same_generation_limit=True,same_precision_backend=True,
                teacher_answer_in_student_prompt=False,reference_answers_used=False)


@contextmanager
def float32_cartridge(*,matched_controls=False):
    """Materialize only the independent selected Value cache in float32.

    The patched hooks are process-global during this context, so it must not
    be used concurrently with other requests. The normal runtime must release
    the full cache, selected source and training adapter before decoding.
    """
    import masked_ttt.pipeline as pipeline
    from masked_ttt.standalone import StandaloneReady

    original_answer=pipeline.answer_standalone
    original_decode=pipeline.decode_standalone

    def prepare(engine,layout,selected,teacher,trained,limit):
        controls=(_paired_precision_controls(engine,layout,selected,teacher,trained,limit)
                  if matched_controls else None)
        with _float_values_during_materialization():
            result=original_answer(engine,layout,selected,teacher,trained,limit)
        if not isinstance(result,StandaloneReady):
            raise RuntimeError('A deferred standalone decode is required')
        result._float32_matched_controls=controls
        return result

    def decode(engine,ready):
        value_dtypes=sorted({str(item.values.dtype) for item in ready.branch.layers})
        with _mixed_precision_attention() as statistics:
            result=original_decode(engine,ready)
        result['memory']['experimental_value_dtypes']=value_dtypes
        result['memory']['experimental_materialization']=(
            'float32_value_cache' if 'torch.float32' in value_dtypes else 'native_no_active_value_update')
        result['memory']['experimental_float_attention_calls']=statistics['float32_value_attention_calls']
        controls=ready._float32_matched_controls
        if controls is not None:
            if controls['adapted']['tokens']!=result['generated_token_ids']:
                raise AssertionError('Precision control and formal standalone answer differ')
            result['precision_matched_controls']=controls
        return result

    pipeline.answer_standalone=prepare
    pipeline.decode_standalone=decode
    try:yield
    finally:
        pipeline.answer_standalone=original_answer
        pipeline.decode_standalone=original_decode
