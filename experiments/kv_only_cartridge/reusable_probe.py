"""Check reusable selected KV against a frozen runtime on an allocated model.

Only synthetic text is used. The candidate modules are temporary, and model
configuration and random states are restored before the benchmark resumes.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

from experiments.kv_only_cartridge.in_job_probe import candidate_pipeline, capture_decode_logits


@contextmanager
def candidate_cartridge(path):
    name = "masked_ttt.cartridge"
    previous = sys.modules.get(name)
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise ValueError("The candidate cartridge snapshot is required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@contextmanager
def capture_answer_logits(engine):
    logits = []
    handle = engine.model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: logits.append(output.detach().float().cpu().clone()))
    try:
        yield logits
    finally:
        handle.remove()


def validate_loaded_reusable(engine, pipeline_path, cartridge_path):
    """Compare first and repeated decoding without altering the benchmark runtime."""
    import masked_ttt.pipeline as frozen_pipeline
    from masked_ttt import Request
    from masked_ttt.pipeline import MaskedTTT
    from masked_ttt.memory_accounting import module_tensors
    from masked_ttt.request import prepare_layout
    from masked_ttt.selection import PreparedRequest

    config, cpu_rng = engine.config, torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state_all() if engine.device.type == "cuda" else None
    frozen_hooks = (frozen_pipeline.answer_standalone, frozen_pipeline.decode_standalone)
    artifact = None
    try:
        engine.config = replace(config, chunk_tokens=64, steps=2, max_answer_steps=2,
            self_question_count=0, student_rollout_tokens=3, guard_enabled=False,
            standalone_cartridge_enabled=True, standalone_answer_format="original",
            selected_transfer_weight=0., value_basis_source="document", align_student_teacher_prompt=False,
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        request = Request(document="The old note is red. " + "Ordinary context. " * 30 +
                          "The current note is blue.", question="What color is the current note?", max_new_tokens=4)
        class Selector:
            def prepare(self, backbone, item):
                layout = prepare_layout(backbone, item)
                return PreparedRequest(item, layout, [len(layout.chunks)//2], dict(backend="synthetic"))
        scope = getattr(engine.tokenizer, "request_scope", None)
        with (scope() if callable(scope) else nullcontext()), candidate_cartridge(cartridge_path), \
             candidate_pipeline(pipeline_path) as candidate:
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:
                torch.cuda.set_rng_state_all(gpu_rng)
            with capture_decode_logits(frozen_pipeline) as baseline_logits:
                baseline = MaskedTTT(engine, Selector()).answer(request, time_budget_seconds=300,
                                                               compact_standalone=True)
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:
                torch.cuda.set_rng_state_all(gpu_rng)
            with capture_decode_logits(candidate) as first_logits:
                artifact, first = candidate.MaskedTTT(engine, Selector()).prepare_cartridge(
                    request, time_budget_seconds=300)
            with capture_answer_logits(engine) as replay_logits:
                replay = artifact.answer(time_budget_seconds=300)
            second = artifact.answer(time_budget_seconds=300)
            traces = (baseline_logits, first_logits, replay_logits)
            if (not baseline_logits or not all(len(trace) == len(baseline_logits) for trace in traces) or
                any(a.shape != b.shape or not torch.allclose(a, b, atol=1e-5, rtol=1e-5)
                    for trace in traces[1:] for a, b in zip(baseline_logits, trace)) or
                baseline["generated_token_ids"] != first["generated_token_ids"] or
                first["generated_token_ids"] != replay["generated_token_ids"] or
                replay["generated_token_ids"] != second["generated_token_ids"] or
                baseline["teacher"]["tokens"] != first["teacher"]["tokens"] or
                baseline["request_chat_date"] != first["request_chat_date"]):
                raise AssertionError("The reusable output or logits changed")
            memory = first["standalone_cartridge"]["memory"]
            if (not first["backbone_unchanged"] or not first["original_document_KV_unchanged"] or
                first["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0 or
                not memory["kv_only_materialized"] or memory["additional_source_kv_bytes_before_decode"] != 0 or
                memory["additional_training_adapter_bytes_before_decode"] != 0 or module_tensors(artifact._reader.memory) or
                artifact.document_kv_bytes != memory["materialized_selected_kv_bytes"]):
                raise AssertionError("The reusable cartridge retained source KV or adapter state")
            if (frozen_pipeline.answer_standalone, frozen_pipeline.decode_standalone) != frozen_hooks:
                raise AssertionError("The benchmark runtime hooks were changed")
            return dict(passed=True, same_first_tokens=True, same_teacher_tokens=True,
                same_first_and_replay_logits=True, repeatable_original_question=True,
                full_and_source_kv_released=True, adapter_tensors_retained=False,
                model_type=engine.model.config.model_type, reference_answers_used=False, online_api_calls=0)
    finally:
        if artifact is not None:
            artifact.close()
        engine.config = config
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state_all(gpu_rng)


def run_if_needed(engine, model, identity):
    """Run at most once per backbone inside an existing allocation."""
    if model not in ("qwen", "gpt") or engine.device.type != "cuda":
        raise ValueError("A supported backbone on an allocated GPU is required")
    root = Path(__file__).resolve().parent
    with (root / (model + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / (model + "_result.json")
        if path.exists() and json.loads(path.read_text()).get("passed"):
            return "previously_passed"
        failure = root / (model + "_failure.json")
        attempts = json.loads(failure.read_text()).get("attempts", 0) if failure.exists() else 0
        if attempts >= 2:
            return "two_previous_failures"
        try:
            data = validate_loaded_reusable(engine, root / "pipeline_candidate.py", root / "cartridge_candidate.py")
            data.update(gpu=identity.get("gpu"), model=model, timestamp=time.time())
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n")
            temporary.replace(path)
            return "passed"
        except Exception as exc:
            temporary = failure.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(attempts=attempts+1, error_type=type(exc).__name__,
                                                 timestamp=time.time()), indent=2) + "\n")
            temporary.replace(failure)
            raise
