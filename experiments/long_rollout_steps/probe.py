"""Read-only 2/4/8-base-step snapshots from one on-policy SD trajectory."""
from contextlib import contextmanager
import weakref

import torch


@torch.no_grad()
def decode(engine,reader,document_cache,suffix,length,limit):
    from masked_ttt.runtime import private_forward
    reader.memory.select_actual_read_path()
    branch=reader.compiled_branch(document_cache)
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
def capture_step_snapshots(base_steps=(2,4,8), *, final_base_step=None):
    """Decode checkpoints on the same independent selected KV after full release.

    An original request uses base steps plus one step per verified practice.
    For plain BABILONG questions, the first 2+P updates of an 8+P run match
    the default 2+P run. The probe never uses reference answers.

    A caller may fix an earlier base step before the request. In that case
    the final, physically released cartridge is materialized from exactly
    that step's state, even though the diagnostic trajectory runs longer.
    The selection itself is an experimental configuration; it is never made
    from the request's generated answers or post-generation labels here.
    """
    import masked_ttt.pipeline as pipeline
    from masked_ttt.memory import KVReadMemory
    from masked_ttt.memory_accounting import storage_bytes
    from masked_ttt.standalone import build_standalone
    base_steps=tuple(base_steps)
    if (not base_steps or any(type(step) is not int or step < 1 for step in base_steps)
            or tuple(sorted(set(base_steps))) != base_steps):
        raise ValueError('Base step budgets must be sorted unique positive integers')
    if final_base_step is None:
        final_base_step=base_steps[-1]
    if type(final_base_step) is not int or final_base_step not in base_steps:
        raise ValueError('The fixed final step must be one of the recorded steps')
    old_reader=pipeline.MemoryReader
    old_study=pipeline.generate_self_study
    old_constrain=KVReadMemory.constrain
    old_answer=pipeline.answer_standalone
    old_decode=getattr(pipeline,'decode_standalone',None)
    captured=dict(snapshots={},count=0,practice_count=None)
    memory_ref=None

    def reader(engine,memory):
        nonlocal memory_ref
        value=old_reader(engine,memory)
        if memory_ref is None:memory_ref=weakref.ref(memory)
        return value

    def study(*args,**kwargs):
        if len(args)>3 and 'Choices:' in args[3].question:
            raise RuntimeError('Step probe is intended for non-choice BABILONG questions')
        try:paths,metadata=old_study(*args,**kwargs)
        except ValueError:
            captured['practice_count']=0
            raise
        captured['practice_count']=len(paths)
        return paths,metadata

    def constrain(memory):
        result=old_constrain(memory)
        if memory_ref is not None and memory_ref() is memory:
            if captured['practice_count'] is None:raise AssertionError('Practice count not established')
            captured['count']+=1
            for base in base_steps:
                if captured['count']==base+captured['practice_count']:
                    snapshot=memory.snapshot()
                    # Checkpoint diagnostics must not retain training tensors
                    # on the GPU during final standalone decoding.
                    captured['snapshots'][base]={key:(value.cpu() if isinstance(value,torch.Tensor) else value)
                                                 for key,value in snapshot.items()}
        return result

    @torch.no_grad()
    def answer(engine,layout,selected,teacher,trained,limit):
        if set(captured['snapshots'])!=set(base_steps):
            raise AssertionError('One or more requested step snapshots are missing')
        saved=trained.snapshot();checkpoints={};completed=False
        snapshot_bytes=storage_bytes(
            [value for snapshot in captured['snapshots'].values() for value in snapshot.values()
             if isinstance(value,torch.Tensor)])
        try:
            for base in base_steps:
                trained.restore(captured['snapshots'][base])
                reader,branch,suffix,length,_=build_standalone(
                    engine,layout,selected,teacher,trained)
                del branch
                output=decode(engine,reader,teacher.document_cache,suffix,length,limit)
                checkpoints[str(base)]=dict(base_steps=base,
                    total_steps=base+captured['practice_count'],
                    prediction=output['prediction'],tokens=output['tokens'])
            # The final release path must use the predeclared checkpoint,
            # not merely report its output while serving the last update.
            trained.restore(captured['snapshots'][final_base_step])
            completed=True
        finally:
            if not completed:
                trained.restore(saved)
            del saved
            # No checkpoint or paired-source tensors are carried into final decode.
            captured['snapshots'].clear()
        del reader,suffix,output
        diagnostics=dict(checkpoints=checkpoints,practice_count=captured['practice_count'],
            fixed_final_base_step=final_base_step,
            same_selected_kv=True,full_kv_used_at_decode=False,
            reference_answers_used=False,teacher_answer_in_final_prompt=False,
            diagnostic_snapshot_bytes=snapshot_bytes,diagnostic_snapshots_offloaded_to_cpu=True,
            diagnostic_snapshots_released_before_final_decode=True)
        captured['diagnostics']=diagnostics
        final=old_answer(engine,layout,selected,teacher,trained,limit)
        if isinstance(final,dict):final['step_probe']=diagnostics
        return final

    def final_decode(engine,ready):
        result=old_decode(engine,ready)
        result['step_probe']=captured['diagnostics']
        return result

    pipeline.MemoryReader=reader;pipeline.generate_self_study=study
    KVReadMemory.constrain=constrain;pipeline.answer_standalone=answer
    if old_decode is not None:pipeline.decode_standalone=final_decode
    try:yield captured
    finally:
        pipeline.MemoryReader=old_reader;pipeline.generate_self_study=old_study
        KVReadMemory.constrain=old_constrain;pipeline.answer_standalone=old_answer
        if old_decode is not None:pipeline.decode_standalone=old_decode
        captured['snapshots'].clear()
