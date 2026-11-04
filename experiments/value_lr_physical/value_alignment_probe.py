"""Compare the same selected tokens' Value updates in two independent caches."""
from contextlib import contextmanager
import math
import statistics
from types import SimpleNamespace

import torch


def _median(values):
    return statistics.median(values.detach().float().cpu().tolist()) if values.numel() else None


def _cosine(left, right):
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    valid = (left_norm > 1e-12) & (right_norm > 1e-12)
    score = ((left * right).sum(dim=-1) / (left_norm * right_norm).clamp_min(1e-30)).clamp(-1., 1.)
    return _median(score[valid]), int(valid.sum().item())


@torch.no_grad()
def _sampled_write(native, basis, coefficients, bank_list, radius):
    """Use the production writer with selected positions packed by bank."""
    from masked_ttt.memory import KVReadMemory
    device = native.device
    chunks = [[i for i, bank in enumerate(bank_list) if bank == c] for c in range(coefficients.shape[1])]
    width = max(map(len, chunks))
    positions = torch.full((len(chunks), width), -1, dtype=torch.long, device=device)
    slots = []
    for bank, indices in enumerate(chunks):
        if indices:
            positions[bank, :len(indices)] = torch.tensor(indices, dtype=torch.long, device=device)
            slots.extend(bank * width + i for i in range(len(indices)))
    state = SimpleNamespace(document_length=len(bank_list), source_positions=tuple(range(len(bank_list))),
        source_banks=tuple(bank_list), bank_positions=positions,
        bank_scatter_slots=torch.tensor(slots, dtype=torch.long, device=device),
        bank_source_indices=torch.arange(len(bank_list), dtype=torch.long, device=device),
        value_basis=basis.unsqueeze(0), value_coefficients=coefficients.unsqueeze(0), value_radius=radius)
    return KVReadMemory._chunk_values(state, 0, native.unsqueeze(0))[0]


@torch.no_grad()
def compare_value_samples(full, selected, basis, coefficients, banks, radius):
    """Inspect real paired cache vectors and sampled post-rounding writes.

    `full` and `selected` are [KV heads, positions, head_dim]; `banks` maps
    positions to the same original selected chunks used by the trained state.
    """
    if (full.ndim != 3 or full.shape != selected.shape or full.dtype != selected.dtype or
        not full.dtype.is_floating_point or not basis.dtype.is_floating_point or
        not coefficients.dtype.is_floating_point or banks.dtype != torch.long or
        basis.ndim != 3 or coefficients.ndim != 4 or banks.ndim != 1 or
        full.shape[0] != basis.shape[0] or full.shape[0] != coefficients.shape[0] or
        full.shape[-1] != basis.shape[-2] or full.shape[-1] != coefficients.shape[-1] or
        basis.shape[-1] != coefficients.shape[-2] or banks.numel() != full.shape[1] or
        not full.shape[1] or not 0 < radius <= 1):
        raise ValueError("Incompatible Value sample, basis or coefficient shapes")
    device = full.device
    if any(x.device != device for x in (selected, basis, coefficients, banks)):
        raise ValueError("Value diagnostic tensors must share a device")
    if bool((banks < 0).any()) or bool((banks >= coefficients.shape[1]).any()):
        raise ValueError("Selected-token bank is out of range")
    bank_list = banks.detach().cpu().tolist()
    if bank_list != sorted(bank_list):
        raise ValueError("Selected-token banks must be in document order")
    x, y = full.float(), selected.float()
    a = basis.float()
    zx = torch.bmm(x, a)
    zy = torch.bmm(y, a)

    # Sampling can change batch shapes, so this is not a promise of bitwise
    # equality to every device's full-cache kernel.
    dx = _sampled_write(full, basis, coefficients, bank_list, radius).float() - x
    dy = _sampled_write(selected, basis, coefficients, bank_list, radius).float() - y
    if any(not bool(torch.isfinite(value).all()) for value in (x, y, zx, zy, dx, dy)):
        raise FloatingPointError("Nonfinite Value transfer diagnostic")
    value_cosine, value_pairs = _cosine(x, y)
    latent_cosine, latent_pairs = _cosine(zx, zy)
    delta_cosine, delta_pairs = _cosine(dx, dy)
    full_denominator = x.square().sum(dim=-1)
    selected_denominator = y.square().sum(dim=-1)
    full_energy = zx.square().sum(dim=-1) / full_denominator.clamp_min(1e-30)
    selected_energy = zy.square().sum(dim=-1) / selected_denominator.clamp_min(1e-30)
    def relative(delta, native):
        denominator = float(native.square().sum())
        return math.sqrt(float(delta.square().sum()) / denominator) if denominator > 1e-30 else None
    return dict(
        sampled_tokens=full.shape[1], kv_heads=full.shape[0],
        median_native_cosine=value_cosine, native_cosine_pairs=value_pairs,
        median_basis_coordinate_cosine=latent_cosine, basis_cosine_pairs=latent_pairs,
        median_written_delta_cosine=delta_cosine, written_delta_cosine_pairs=delta_pairs,
        median_full_basis_energy=_median(full_energy[full_denominator > 1e-30]),
        median_selected_basis_energy=_median(selected_energy[selected_denominator > 1e-30]),
        full_written_relative_l2=relative(dx, x), selected_written_relative_l2=relative(dy, y),
        full_changed_fraction=float((dx != 0).float().mean()),
        selected_changed_fraction=float((dy != 0).float().mean()),
    )


def selected_token_pairs(engine, layout, selected, teacher, memory, max_positions=256):
    """Align the original token and independently-prefetched selected token."""
    from masked_ttt._layout import selected_read_plan
    from masked_ttt.standalone import selected_chunk_positions

    if type(max_positions) is not int or max_positions < 1 or tuple(selected) != memory.selected_chunks:
        raise ValueError("Invalid selection or sampling budget")
    separator = engine.tokenizer.encode("\n\n", add_special_tokens=False)
    plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, selected, separator)
    if tuple(plan.ids[:plan.document_end]) != teacher.document_ids:
        raise ValueError("Selected teacher cache does not match the token plan")
    compact = selected_chunk_positions(layout, selected, plan, len(separator))
    active = set(memory.source_positions)
    positions = []
    for bank, original in enumerate(selected):
        for src, dst in zip(layout.chunks[original], compact[bank]):
            if src not in active or dst < engine.config.sink_tokens:
                continue
            if layout.ids[src] != plan.ids[dst]:
                raise ValueError("The selected token identity changed across caches")
            positions.append((src, dst, bank))
    if not positions:
        raise ValueError("No addressable selected Value positions")
    count = min(max_positions, len(positions))
    chosen = [positions[(i * (len(positions) - 1)) // max(1, count - 1)] for i in range(count)]
    return positions, chosen


@torch.no_grad()
def inspect_value_alignment(engine, layout, selected, teacher, full_cache, memory, max_positions=256):
    positions, chosen = selected_token_pairs(engine, layout, selected, teacher, memory, max_positions)
    full_pos, selected_pos, banks = zip(*chosen)
    versions = [(value, value._version, value.data_ptr()) for value in
                (memory.gain, memory.value_basis, memory.value_coefficients)]
    rows = []
    for layer in memory.adapted_layers:
        if layer not in memory.active_value_layers:
            continue
        x = full_cache.layers[layer].values
        y = teacher.document_cache.layers[layer].values
        versions.extend(((x, x._version, x.data_ptr()), (y, y._version, y.data_ptr())))
        if x.device != y.device:
            raise ValueError("Paired caches must be on the same device")
        xi = torch.as_tensor(full_pos, dtype=torch.long, device=x.device)
        yi = torch.as_tensor(selected_pos, dtype=torch.long, device=x.device)
        bi = torch.as_tensor(banks, dtype=torch.long, device=x.device)
        row = compare_value_samples(x[0].index_select(1, xi), y[0].index_select(1, yi),
            memory.value_basis[layer].to(x.device), memory.value_coefficients[layer].to(x.device), bi,
            memory.value_radius)
        row["layer"] = int(layer)
        rows.append(row)
    if not rows:
        raise ValueError("No active Value layers")
    if any(value._version != version or value.data_ptr() != pointer for value, version, pointer in versions):
        raise AssertionError("An original cache or trained adapter was modified")
    def layer_median(key):
        values = [row[key] for row in rows if row[key] is not None]
        return statistics.median(values) if values else None
    return dict(selected_tokens=len(positions), sampled_tokens=len(chosen), layers=rows,
        median_layer_native_cosine=layer_median("median_native_cosine"),
        median_layer_basis_coordinate_cosine=layer_median("median_basis_coordinate_cosine"),
        median_layer_written_delta_cosine=layer_median("median_written_delta_cosine"),
        reference_answers_used=False, model_or_cache_modified=False)


@contextmanager
def capture_value_alignment(max_positions=256):
    """Observe after SD, then release every cache and model reference."""
    import masked_ttt.pipeline as pipeline
    from masked_ttt.memory import KVReadMemory

    old_prefill, old_teacher, old_reader = pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader
    old_select = KVReadMemory.select_actual_read_path
    captured = {}

    def prefill(*args, **kwargs):
        cache = old_prefill(*args, **kwargs)
        captured["_full_cache"] = cache
        return cache

    def teacher(engine, reader, request, layout, selected):
        result = old_teacher(engine, reader, request, layout, selected)
        captured.update(_engine=engine, _layout=layout, _selected=list(selected), _teacher=result)
        return result

    def reader(engine, memory):
        result = old_reader(engine, memory)
        if "_full_reader" not in captured:
            captured["_full_reader"] = result
        return result

    def clear_tensors():
        for key in tuple(captured):
            if key.startswith("_"):
                captured.pop(key, None)

    def select(memory):
        old_select(memory)
        full_reader, cache = captured.get("_full_reader"), captured.get("_full_cache")
        if full_reader is None or full_reader.memory is not memory or cache is None:
            return
        try:
            captured["statistics"] = inspect_value_alignment(captured["_engine"], captured["_layout"],
                captured["_selected"], captured["_teacher"], cache, memory, max_positions)
        except (ValueError, RuntimeError, FloatingPointError, IndexError, TypeError) as exc:
            captured["statistics"] = dict(unavailable=type(exc).__name__, reason=str(exc),
                reference_answers_used=False, model_or_cache_modified=False)
        finally:
            clear_tensors()

    pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader = prefill, teacher, reader
    KVReadMemory.select_actual_read_path = select
    try:
        yield captured
    finally:
        pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader = old_prefill, old_teacher, old_reader
        KVReadMemory.select_actual_read_path = old_select
        clear_tensors()
