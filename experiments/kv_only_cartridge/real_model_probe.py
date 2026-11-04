"""Check KV-only decode against ordinary decoding on an already-loaded model."""
from contextlib import contextmanager
from dataclasses import replace
import math

import torch

from experiments.kv_only_cartridge.compact import kv_only_cartridge
from masked_ttt import MaskedTTT, Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest


def _long_attention_probe(engine):
    """Verify 32-position GQA/sink attention and both gradients on this GPU."""
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    from masked_ttt.attention import address_attention

    config = engine.model.config
    query_heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim = getattr(config, "head_dim", config.hidden_size // query_heads)
    device, dtype = engine.device, torch.bfloat16
    query = torch.randn(1, query_heads, 32, dim, device=device, dtype=dtype)
    key = torch.randn(1, kv_heads, 64, dim, device=device, dtype=dtype)
    value = torch.randn(1, kv_heads, 64, dim, device=device, dtype=dtype, requires_grad=True)
    gain = (torch.randn(kv_heads, 64, device=device, dtype=torch.float32) * .3).requires_grad_()
    sinks = (torch.randn(query_heads, device=device, dtype=torch.float32) * .2
             if config.model_type == "gpt_oss" else None)
    probe = torch.randn(1, query_heads, 32, dim, device=device, dtype=torch.float32)
    backend = "flash" if device.type == "cuda" else "eager"
    observed = address_attention(query, key, value, gain, backend=backend, sinks=sinks)
    got_gain, got_value = torch.autograd.grad((observed.float() * probe).sum(), (gain, value))

    reference_gain = gain.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    groups = query_heads // kv_heads
    scores = query.float() @ repeat_kv(key.float(), groups).transpose(-1, -2) / math.sqrt(dim)
    scores = scores + reference_gain.repeat_interleave(groups, 0)[None, :, None, :]
    visible = torch.arange(64, device=device)[None] <= torch.arange(32, 64, device=device)[:, None]
    scores = scores.masked_fill(~visible[None, None], float("-inf"))
    if sinks is not None:
        scores = torch.cat((scores, sinks[None, :, None, None].expand(1, -1, 32, -1)), -1)
    probabilities = scores.softmax(-1)
    if sinks is not None:
        probabilities = probabilities[..., :-1]
    expected = probabilities @ repeat_kv(reference_value.float(), groups)
    want_gain, want_value = torch.autograd.grad(
        (expected * probe).sum(), (reference_gain, reference_value)
    )
    torch.testing.assert_close(observed.float(), expected, atol=.018, rtol=.04)
    torch.testing.assert_close(got_gain, want_gain, atol=.08, rtol=.08)
    torch.testing.assert_close(got_value.float(), want_value.float(), atol=.035, rtol=.05)
    return backend


@contextmanager
def _capture_standalone_logits():
    import masked_ttt.pipeline as pipeline

    original = pipeline.decode_standalone
    logits = []

    def decode(engine, ready):
        def record(module, inputs, output):
            logits.append(output.detach().float().cpu().clone())

        hook = engine.model.lm_head.register_forward_hook(record)
        try:
            return original(engine, ready)
        finally:
            hook.remove()

    pipeline.decode_standalone = decode
    try:
        yield logits
    finally:
        pipeline.decode_standalone = original


def validate_loaded_backbone(engine):
    """Run a synthetic paired request without benchmark data or extra models."""
    old_config = engine.config
    try:
        long_attention_backend = _long_attention_probe(engine)
        engine.config = replace(
            old_config,
            chunk_tokens=64,
            steps=2,
            max_answer_steps=2,
            self_question_count=0,
            student_rollout_tokens=3,
            guard_enabled=False,
            standalone_cartridge_enabled=True,
            standalone_answer_format="original",
            selected_transfer_weight=0.0,
            value_basis_source="document",
            release_full_before_cartridge=True,
            release_selected_source_before_decode=True,
        )
        request = Request(
            document="The old note is red. " + "Ordinary context. " * 30
            + "The current note is blue.",
            question="What color is the current note?",
            max_new_tokens=4,
        )

        class Selector:
            def prepare(self, backbone, item):
                layout = prepare_layout(backbone, item)
                return PreparedRequest(item, layout, [len(layout.chunks) // 2],
                                       dict(backend="synthetic_kv_only_check"))

        method = MaskedTTT(engine, Selector())
        with _capture_standalone_logits() as reference_logits:
            reference = method.answer(request, time_budget_seconds=300)
        with _capture_standalone_logits() as actual_logits, kv_only_cartridge():
            actual = method.answer(request, time_budget_seconds=300)

        before = reference["standalone_cartridge"]["memory"]
        after = actual["standalone_cartridge"]["memory"]
        steps = actual["training"]["steps"]
        same_logits = (
            len(reference_logits) == len(actual_logits)
            and len(actual_logits) > 0
            and all(x.shape == y.shape and torch.allclose(x, y, atol=1e-5, rtol=1e-5)
                    for x, y in zip(reference_logits, actual_logits))
        )
        if (
            not same_logits
            or actual["generated_token_ids"] != reference["generated_token_ids"]
            or actual["teacher"]["tokens"] != reference["teacher"]["tokens"]
            or not actual["backbone_unchanged"]
            or not actual["original_document_KV_unchanged"]
            or not any(step["gain_gradient_norm"] > 0 for step in steps)
            or not any(step["value_gradient_norm"] > 0 for step in steps)
            or actual["memory"]["full_cache_release_before_standalone"][
                "full_cache_tensors_live_before_standalone"
            ] != 0
            or after["additional_source_kv_bytes_before_decode"] != 0
            or after["additional_training_adapter_bytes_before_decode"] != 0
            or after["cartridge_adapter_bytes"] != 0
            or after["standalone_cartridge_bytes"] >= before["standalone_cartridge_bytes"]
        ):
            raise AssertionError("The KV-only cartridge failed the paired model check")
        return dict(
            passed=True,
            same_teacher_tokens=True,
            same_standalone_tokens=True,
            same_standalone_logits=True,
            long_attention_positions=32,
            long_attention_backend=long_attention_backend,
            long_attention_gradients_match=True,
            evaluated_positions=len(actual_logits),
            source_and_training_released=True,
            adapter_bytes_before=before["cartridge_adapter_bytes"],
            adapter_bytes_after=after["cartridge_adapter_bytes"],
            cartridge_bytes_before=before["standalone_cartridge_bytes"],
            cartridge_bytes_after=after["standalone_cartridge_bytes"],
            online_api_calls=0,
            reference_answers_used=False,
        )
    finally:
        engine.config = old_config
