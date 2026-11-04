"""Verify a request-local cartridge in an already-allocated model process.

The frozen benchmark runtime stays untouched. Only the candidate pipeline is
loaded under a private module name, while its supporting modules and the
already-loaded backbone come from the benchmark. The optional driver restores
the engine configuration and CPU/GPU random states before evaluation resumes.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys

import torch


@contextmanager
def candidate_pipeline(path):
    name = "masked_ttt._request_local_validation_pipeline"
    if name in sys.modules:
        raise RuntimeError("A request-local validation is already active")
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise ValueError("The frozen candidate pipeline is required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@contextmanager
def capture_decode_logits(pipeline):
    original = pipeline.decode_standalone
    logits = []
    def decode(engine, ready):
        hook = engine.model.lm_head.register_forward_hook(
            lambda _module, _inputs, output: logits.append(output.detach().float().cpu().clone()))
        try:
            return original(engine, ready)
        finally:
            hook.remove()
    pipeline.decode_standalone = decode
    try:
        yield logits
    finally:
        pipeline.decode_standalone = original


def validate_loaded_candidate(engine, candidate_path):
    """Compare frozen and candidate decode paths on one synthetic request."""
    import masked_ttt.pipeline as frozen_pipeline
    from masked_ttt import MaskedTTT, Request
    from masked_ttt.kv_only import kv_only_cartridge
    from masked_ttt.request import prepare_layout
    from masked_ttt.selection import PreparedRequest

    frozen_entrypoints = (frozen_pipeline.answer_standalone, frozen_pipeline.decode_standalone)
    original_config = engine.config
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state_all() if engine.device.type == "cuda" else None
    try:
        engine.config = replace(original_config, chunk_tokens=64, steps=2, max_answer_steps=2,
            self_question_count=0, student_rollout_tokens=3, guard_enabled=False,
            standalone_cartridge_enabled=True, standalone_answer_format="original",
            selected_transfer_weight=0., value_basis_source="document",
            align_student_teacher_prompt=False,
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        request = Request(document="The old note is red. " + "Ordinary context. " * 30 +
                          "The current note is blue.", question="What color is the current note?",
                          max_new_tokens=4)

        class Selector:
            def prepare(self, backbone, item):
                layout = prepare_layout(backbone, item)
                return PreparedRequest(item, layout, [len(layout.chunks) // 2],
                                       dict(backend="synthetic_request_local_probe"))

        scope = getattr(engine.tokenizer, "request_scope", None)
        # GPT-OSS inserts the current date into its chat template. Pin that
        # context across both paired calls, even if midnight occurs between them.
        with (scope() if callable(scope) else nullcontext()):
            with candidate_pipeline(candidate_path) as candidate:
                torch.set_rng_state(cpu_rng)
                if gpu_rng is not None:
                    torch.cuda.set_rng_state_all(gpu_rng)
                with capture_decode_logits(frozen_pipeline) as old_logits, kv_only_cartridge():
                    old = MaskedTTT(engine, Selector()).answer(request, time_budget_seconds=300)
                # The benchmark's random state and the candidate's own inputs match.
                torch.set_rng_state(cpu_rng)
                if gpu_rng is not None:
                    torch.cuda.set_rng_state_all(gpu_rng)
                with capture_decode_logits(candidate) as new_logits:
                    new = candidate.MaskedTTT(engine, Selector()).answer(
                        request, time_budget_seconds=300, compact_standalone=True)

        if (not old_logits or len(old_logits) != len(new_logits) or
            any(a.shape != b.shape or not torch.allclose(a, b, atol=1e-5, rtol=1e-5)
                for a, b in zip(old_logits, new_logits)) or
            old["generated_token_ids"] != new["generated_token_ids"] or
            old["request_chat_date"] != new["request_chat_date"] or
            old["teacher"]["tokens"] != new["teacher"]["tokens"]):
            raise AssertionError("The request-local output or logits differ from the frozen runtime")
        if (frozen_pipeline.answer_standalone, frozen_pipeline.decode_standalone) != frozen_entrypoints:
            raise AssertionError("The frozen runtime hooks were not restored")
        old_steps, new_steps = old["training"]["steps"], new["training"]["steps"]
        if (len(old_steps) != 2 or len(new_steps) != 2 or
            any(a["student_rollout_tokens"] != b["student_rollout_tokens"] or
                any(not math.isclose(a[k], b[k], rel_tol=1e-5, abs_tol=1e-7)
                    for k in ("teacher_KL_before", "gain_gradient_norm", "value_gradient_norm"))
                for a, b in zip(old_steps, new_steps)) or
            not any(step["gain_gradient_norm"] > 0 for step in new_steps) or
            not any(step["value_gradient_norm"] > 0 for step in new_steps)):
            raise AssertionError("The request-local SD training trace differs")
        for result in (old, new):
            memory = result["standalone_cartridge"]["memory"]
            if (not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
                result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0 or
                not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
                memory["additional_source_kv_bytes_before_decode"] != 0 or
                memory["additional_training_adapter_bytes_before_decode"] != 0):
                raise AssertionError("Source KV or adaptation state survived the independent decode")
        return dict(passed=True, same_teacher_tokens=True, same_student_rollouts=True,
            same_training_gradients=True, same_standalone_tokens=True, same_standalone_logits=True,
            same_request_chat_date=True,
            no_full_kv_or_extra_adapter_at_decode=True, original_benchmark_runtime_unchanged=True,
            model_type=engine.model.config.model_type, online_api_calls=0, reference_answers_used=False)
    finally:
        engine.config = original_config
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state_all(gpu_rng)


def run_if_needed(engine, model, identity):
    """Entry point used by the existing optional-probe hook in a GPU job."""
    from experiments.kv_only_cartridge.driver import run_if_enabled
    directory = Path(__file__).resolve().parent
    return run_if_enabled(engine, model, identity, directory,
        probe=lambda item: validate_loaded_candidate(item, directory / "pipeline_candidate.py"))
