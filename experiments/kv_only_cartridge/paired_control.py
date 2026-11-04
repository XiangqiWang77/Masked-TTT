"""Four prompt controls followed by a physically independent cartridge decode."""
from contextlib import contextmanager

import torch

from experiments.kv_only_cartridge.compact import _drop_redundant_tensors
from experiments.shared_question_sd.controls import _decode
from masked_ttt.standalone import StandaloneReady, build_standalone


@contextmanager
def physical_paired_prompt_control(*, compact=True):
    """Keep diagnostic comparisons separate from the actual final decode.

    Teacher-format and original-format comparisons use the same selected KV,
    as in the historical paired report. Then the ordinary production path
    materializes a fresh cartridge, drops source/training states, and answers
    the original question. The final tokens must equal the trained control.
    Hooks are process-local and must not be shared across concurrent requests.
    """
    import masked_ttt.pipeline as pipeline

    original_answer = pipeline.answer_standalone
    original_decode = pipeline.decode_standalone

    @torch.no_grad()
    def prepare(engine, layout, selected, teacher, trained, limit):
        source = teacher.document_cache
        stamps = [(item, item._version, item.data_ptr()) for layer in source.layers
                  for item in (layer.keys, layer.values) if item is not None]
        reader, branch, original_suffix, length, _ = build_standalone(
            engine, layout, selected, teacher, trained
        )
        del branch
        teacher_suffix = teacher.question_tokens
        if teacher_suffix.ndim != 1 or not len(teacher_suffix):
            raise ValueError("Teacher question suffix is missing")
        if torch.equal(original_suffix, teacher_suffix):
            raise ValueError("The teacher formatting suffix must differ")
        memory = reader.memory
        adapted_original = _decode(engine, reader, source, original_suffix, length, limit)
        adapted_teacher = _decode(engine, reader, source, teacher_suffix, length, limit)
        snapshot = memory.snapshot()
        try:
            memory.gain.zero_()
            memory.value_coefficients.zero_()
            memory.select_actual_read_path()
            if not memory.native_mode:
                raise AssertionError("The control state is not zero")
            zero_original = _decode(engine, reader, source, original_suffix, length, limit)
            zero_teacher = _decode(engine, reader, source, teacher_suffix, length, limit)
        finally:
            memory.restore(snapshot)
        if any(item._version != version or item.data_ptr() != pointer
               for item, version, pointer in stamps):
            raise AssertionError("The selected source cache changed")
        controls = dict(
            trained_original=adapted_original,
            trained_teacher_suffix=adapted_teacher,
            zero_original=zero_original,
            zero_teacher_suffix=zero_teacher,
            same_selected_kv=True,
            same_backbone_and_device=True,
            same_generation_limit=True,
            control_update_steps=0,
            teacher_answer_in_student_prompt=False,
            reference_answers_used=False,
            teacher_suffix_token_count=len(teacher_suffix),
            original_suffix_token_count=len(original_suffix),
        )
        del reader, memory, source, stamps, snapshot, original_suffix, teacher_suffix
        ready = original_answer(engine, layout, selected, teacher, trained, limit)
        if not isinstance(ready, StandaloneReady):
            raise RuntimeError("Deferred independent cartridge decoding is required")
        if compact:
            _drop_redundant_tensors(ready)
        ready._physical_paired_control = controls
        return ready

    def decode(engine, ready):
        controls = ready._physical_paired_control
        result = original_decode(engine, ready)
        if result["generated_token_ids"] != controls["trained_original"]["tokens"]:
            raise AssertionError("The independent answer differs from its matched control")
        memory = result["memory"]
        if (memory["additional_source_kv_bytes_before_decode"] != 0 or
            memory["additional_training_adapter_bytes_before_decode"] != 0):
            raise AssertionError("Source or training state remained at independent decode")
        if compact and (memory["cartridge_adapter_bytes"] != 0 or
                        not memory["kv_only_materialized"]):
            raise AssertionError("The final cartridge retained redundant adapter tensors")
        result["paired_prompt_control"] = controls
        return result

    pipeline.answer_standalone = prepare
    pipeline.decode_standalone = decode
    try:
        yield
    finally:
        pipeline.answer_standalone = original_answer
        pipeline.decode_standalone = original_decode
