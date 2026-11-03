"""Measure live KV and cartridge tensor storage without double-counting views."""

import torch


def storage_bytes(tensors):
    """Count backing storage once per device, including reserved view capacity."""
    seen=set();total=0
    for tensor in tensors:
        if not isinstance(tensor,torch.Tensor):
            continue
        storage=tensor.untyped_storage()
        if storage.nbytes()==0:
            continue
        key=(str(tensor.device),storage.data_ptr())
        if key not in seen:
            seen.add(key);total+=storage.nbytes()
    return total


def cache_tensors(cache):
    return [tensor for layer in cache.layers for tensor in (layer.keys,layer.values)
            if isinstance(tensor,torch.Tensor)]


def module_tensors(module):
    return list(module.parameters())+list(module.buffers())


def optimizer_storage_bytes(optimizer):
    return storage_bytes(value for state in optimizer.state.values() for value in state.values())


def cuda_allocated(device):
    return int(torch.cuda.memory_allocated(device)) if device.type=='cuda' else None


def cuda_peak(device):
    return int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None
