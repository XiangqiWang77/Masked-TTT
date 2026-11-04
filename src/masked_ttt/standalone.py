"""Build a self-contained selected-KV cartridge from one trained request."""

import time
import torch

from ._layout import selected_read_plan
from .memory import KVReadMemory
from .memory_accounting import cache_tensors,module_tensors,storage_bytes
from .reader import MemoryReader
from .runtime import private_forward


def selected_chunk_positions(layout,selected,plan,separator_length):
    """Map original selected token slots to the selected-only cache slots."""
    if separator_length<1:
        raise ValueError('A nonempty separator is required')
    mapping={};cursor=layout.initial;previous=layout.initial-1
    for original in plan.original_document_positions:
        if original!=previous+1:
            cursor+=separator_length
        mapping[original]=cursor;cursor+=1;previous=original
    if cursor!=plan.document_end:
        raise ValueError('Selected cache position mapping is inconsistent')
    try:
        return [[mapping[position] for position in layout.chunks[i]] for i in selected]
    except KeyError as exc:
        raise ValueError('Selected cache is missing a selected token') from exc


@torch.no_grad()
def build_standalone(engine,layout,selected,teacher,trained):
    """Bake selected-only KV plus the trained state; no full-document KV is used."""
    separator=engine.tokenizer.encode('\n\n',add_special_tokens=False)
    plan=selected_read_plan(layout.ids,layout.chunks,layout.initial,selected,separator)
    if (plan.document_end!=teacher.document_end or
        tuple(plan.ids[:plan.document_end])!=teacher.document_ids):
        raise ValueError('The selected-only KV does not match the cartridge source')
    chunks=selected_chunk_positions(layout,selected,plan,len(separator))
    memory=KVReadMemory(engine.model.config,chunks,list(range(len(chunks))),plan.document_end,
        rank=trained.value_basis.shape[-1],value_radius=trained.value_radius,
        gain_bound=trained.gain_bound,sink_tokens=engine.config.sink_tokens,
        seed=engine.config.seed).to(engine.device)
    memory.gain.copy_(trained.gain[...,selected])
    memory.value_coefficients.copy_(trained.value_coefficients)
    memory.value_basis.copy_(trained.value_basis)
    memory.value_layers.copy_(trained.value_layers)
    memory.active_value_layers=set(trained.active_value_layers)
    memory.value_enabled=trained.value_enabled
    memory.native_mode=trained.native_mode
    for parameter in memory.parameters():parameter.requires_grad_(False)
    reader=MemoryReader(engine,memory)
    branch=reader.compiled_branch(teacher.document_cache)
    suffix=torch.tensor(plan.ids[plan.document_end:],device=engine.device)
    if not len(suffix):raise ValueError('Standalone question suffix is empty')
    accounting=dict(
        source_selected_kv_bytes=storage_bytes(cache_tensors(teacher.document_cache)),
        materialized_selected_kv_bytes=storage_bytes(cache_tensors(branch)),
        cartridge_adapter_bytes=storage_bytes(module_tensors(memory)),
        standalone_cartridge_bytes=storage_bytes(cache_tensors(branch)+module_tensors(memory)),
        document_tokens=plan.document_end,
        selected_chunks=len(chunks),
        raw_full_document_kv_used_at_decode=False,
        selected_kv_representation='independently_prefilled_selected_text',
        includes_generation_suffix_kv=False,
    )
    return reader,branch,suffix,plan.document_end,accounting


@torch.no_grad()
def answer_standalone(engine,layout,selected,teacher,trained,limit):
    began=time.perf_counter()
    reader,branch,suffix,length,accounting=build_standalone(
        engine,layout,selected,teacher,trained)
    if engine.config.standalone_answer_format=='teacher_requirements':
        # Reuse only the question and generic output requirements. Teacher
        # answer tokens are never part of this suffix.
        if teacher.question_tokens is None or teacher.question_tokens.ndim!=1 or not len(teacher.question_tokens):
            raise ValueError('Teacher question requirements are missing')
        suffix=teacher.question_tokens.to(engine.device)
    if limit<1 or length+len(suffix)+limit>engine.config.max_context:
        raise ValueError('Standalone question and generation exceed the model context')
    engine.budget.check()
    output=private_forward(reader.reader,suffix,length,branch)
    tokens=[];first_token_seconds=None
    while len(tokens)<limit:
        token=int(engine.model.lm_head(output.last_hidden_state[0,-1:])[0].argmax())
        tokens.append(token)
        if first_token_seconds is None:
            engine.synchronize()
            first_token_seconds=time.perf_counter()-began
        if token in engine.eos_ids or len(tokens)==limit:break
        engine.budget.check()
        output=private_forward(reader.reader,torch.tensor([token],device=engine.device),
            length+len(suffix)+len(tokens)-1,branch)
    return dict(prediction=engine.tokenizer.decode(tokens,skip_special_tokens=True),
                generated_token_ids=tokens,memory=accounting,
                first_token_seconds=first_token_seconds,
                answer_format=engine.config.standalone_answer_format,
                teacher_answer_in_final_prompt=False)
