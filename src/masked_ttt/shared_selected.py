"""Use one trainable reading state on full and independently prefetched KV."""
import torch

from ._layout import selected_read_plan
from .memory import KVReadMemory
from .reader import MemoryReader
from .standalone import selected_chunk_positions


def shared_selected_reader(engine,layout,selected,teacher,full_memory):
    """Map selected-cache positions onto the original trainable parameter banks.

    The two readers share the actual Parameters, not copies of their values.
    Original source KV caches remain read-only in both paths.
    """
    if tuple(selected)!=full_memory.selected_chunks:
        raise ValueError("The selected chunk order must match the full memory")
    separator=engine.tokenizer.encode("\n\n",add_special_tokens=False)
    plan=selected_read_plan(layout.ids,layout.chunks,layout.initial,selected,separator)
    if (plan.document_end!=teacher.document_end or
        tuple(plan.ids[:plan.document_end])!=teacher.document_ids):
        raise ValueError("Selected teacher cache does not match the source plan")
    chunks=selected_chunk_positions(layout,selected,plan,len(separator))
    view=KVReadMemory(engine.model.config,chunks,list(range(len(chunks))),plan.document_end,
        rank=full_memory.value_basis.shape[-1],value_radius=full_memory.value_radius,
        gain_bound=full_memory.gain_bound,sink_tokens=engine.config.sink_tokens,
        seed=engine.config.seed).to(engine.device)
    mapping=torch.tensor(selected,dtype=torch.long,device=view.chunk_index.device)
    old=view.chunk_index
    view.chunk_index=torch.where(old>=0,mapping[old.clamp_min(0)],old)
    view.selected_chunk_mask=full_memory.selected_chunk_mask
    view.gain=full_memory.gain
    view.value_coefficients=full_memory.value_coefficients
    view.value_basis=full_memory.value_basis
    view.value_layers=full_memory.value_layers
    view.active_value_layers=set(full_memory.active_value_layers)
    view.value_enabled=full_memory.value_enabled
    view.native_mode=False
    if view.gain is not full_memory.gain or view.value_coefficients is not full_memory.value_coefficients:
        raise AssertionError("Both readers must share trainable parameters")
    return MemoryReader(engine,view)
