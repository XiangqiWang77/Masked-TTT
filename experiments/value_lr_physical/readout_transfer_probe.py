"""Read-only first-token comparison with physical release of diagnostic views."""
from contextlib import contextmanager

import torch


@torch.no_grad()
def measure_first_answer(engine, layout, selected, teacher, full_cache, full_reader):
    from types import SimpleNamespace
    from masked_ttt.cache import fork_prefix
    from masked_ttt.runtime import private_forward, private_module_tree
    from masked_ttt.standalone import build_standalone

    native = SimpleNamespace(model=private_module_tree(engine.model.model), lm_head=engine.model.lm_head)
    full_suffix = torch.tensor(layout.ids[layout.document_end:], device=engine.device)
    if not len(full_suffix):
        raise ValueError("Original question suffix is empty")
    full_length = layout.document_end
    selected_reader, selected_branch, selected_suffix, selected_length, _ = build_standalone(
        engine, layout, selected, teacher, full_reader.memory)
    if not torch.equal(full_suffix, selected_suffix):
        raise ValueError("The original question suffix differs across views")

    def logits(reader, cache, suffix, start):
        output = private_forward(reader, suffix, start, cache)
        return engine.model.lm_head(output.last_hidden_state[0, -1:])[0].float()

    d0 = logits(native, fork_prefix(full_cache, full_length), full_suffix, full_length)
    da = logits(full_reader.reader, full_reader.compiled_branch(full_cache), full_suffix, full_length)
    e0 = logits(native, fork_prefix(teacher.document_cache, selected_length), selected_suffix, selected_length)
    ea = logits(selected_reader.reader, selected_branch, selected_suffix, selected_length)
    if any(not bool(torch.isfinite(x).all()) for x in (d0, da, e0, ea)):
        raise FloatingPointError("Nonfinite first-answer distribution")
    ld, la, le, lea = (x.log_softmax(-1) for x in (d0, da, e0, ea))

    def kl(lp, lq):
        return float((lp.exp() * (lp - lq)).sum().clamp_min(0))

    return dict(
        prefix="same original question; before first answer token",
        kl_full_native_to_selected_native=kl(ld, le),
        kl_full_native_to_selected_adapted=kl(ld, lea),
        kl_selected_native_to_selected_adapted=kl(le, lea),
        kl_full_native_to_full_adapted=kl(ld, la),
        full_native_top_token=int(d0.argmax()),
        selected_native_top_token=int(e0.argmax()),
        selected_adapted_top_token=int(ea.argmax()),
        reference_answers_used=False, teacher_answer_in_student_prompt=False,
        full_cache_or_model_modified=False,
    )


@contextmanager
def capture_first_answer_readout():
    """Run the existing readout, then release all objects before final decode.

    This observer patches request-local constructors in a single-request
    process. It stores only scalar statistics after the model views are used.
    """
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
            engine = captured["_engine"]
            if engine.device.type == "cuda" and torch.cuda.get_device_properties(engine.device).total_memory < 70 * 1024 ** 3:
                captured["statistics"] = dict(unavailable="diagnostic_memory_budget", reference_answers_used=False)
            else:
                captured["statistics"] = measure_first_answer(engine, captured["_layout"], captured["_selected"],
                                                               captured["_teacher"], cache, full_reader)
        except (ValueError, RuntimeError, FloatingPointError, IndexError, TypeError) as exc:
            captured["statistics"] = dict(unavailable=type(exc).__name__, reason=str(exc),
                                          reference_answers_used=False)
        finally:
            # Retaining even the reader or Teacher here would keep the
            # training adapter or source-selected cache alive at decode.
            clear_tensors()

    pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader = prefill, teacher, reader
    KVReadMemory.select_actual_read_path = select
    try:
        yield captured
    finally:
        pipeline.prefill_document, pipeline.build_teacher, pipeline.MemoryReader = old_prefill, old_teacher, old_reader
        KVReadMemory.select_actual_read_path = old_select
        clear_tensors()
