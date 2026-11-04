from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import replace

import pytest

from experiments.value_lr_physical.canonical_context import canonical_selected_evidence
from masked_ttt import MaskedTTT, Request
from masked_ttt.evidence import _CURRENT, selected_evidence_mode
from test_canonical_pair import BoundaryTokenizer, Selector
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def setup(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    if architecture == "qwen3":
        engine.tokenizer = BoundaryTokenizer()
    else:
        from masked_ttt.architectures import FinalChannelTokenizer
        class HarmonyTokenizer(BoundaryTokenizer):
            def apply_chat_template(self, messages, **_kwargs):
                return "<|start|>user<|message|>" + messages[0]["content"] + "<|end|><|start|>assistant"
        engine.tokenizer = FinalChannelTokenizer(HarmonyTokenizer())
    engine.config = replace(engine.config, chunk_tokens=32, steps=2, self_question_count=0,
        guard_enabled=False, selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    request = Request(document="A"*32 + "B"*32 + "E"*32 + "F"*32,
                      question="Which block is E?", max_new_tokens=3)
    return engine, request


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_request_local_canonical_matches_completed_experimental_path(architecture):
    engine, request = setup(architecture)
    method = MaskedTTT(engine, Selector())
    with canonical_selected_evidence(engine.tokenizer, sink_tokens=engine.config.sink_tokens):
        expected = method.answer(request, compact_standalone=True)
    engine.config = replace(engine.config, selected_evidence_tokenization="canonical")
    cartridge, actual = method.prepare_cartridge(request)
    try:
        assert actual["selection"]["evidence_tokenization"] == "canonical"
        assert actual["teacher"]["evidence_tokenization"] == "canonical"
        assert actual["teacher"]["prompt_tokens"] == expected["teacher"]["prompt_tokens"]
        assert actual["teacher"]["tokens"] == expected["teacher"]["tokens"]
        assert actual["generated_token_ids"] == expected["generated_token_ids"]
        assert cartridge.answer()["generated_token_ids"] == actual["generated_token_ids"]
        assert len(actual["training"]["steps"]) == len(expected["training"]["steps"])
        for a, b in zip(actual["training"]["steps"], expected["training"]["steps"]):
            assert a["student_rollout_tokens"] == b["student_rollout_tokens"]
            for key in ("teacher_KL_before", "gain_gradient_norm", "value_gradient_norm"):
                assert a[key] == pytest.approx(b[key], abs=1e-7)
        assert actual["backbone_unchanged"] and actual["original_document_KV_unchanged"]
        assert actual["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
        assert actual["standalone_cartridge"]["memory"]["additional_source_kv_bytes_before_decode"] == 0
    finally:
        cartridge.close()
    assert _CURRENT.get() is None


def test_original_and_canonical_requests_are_isolated_between_threads():
    engine, request = setup("qwen3")
    original, canonical = copy(engine), copy(engine)
    canonical.config = replace(engine.config, selected_evidence_tokenization="canonical")
    methods = [MaskedTTT(original, Selector()), MaskedTTT(canonical, Selector())]
    expected = [method.answer(request, compact_standalone=True) for method in methods]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(method.answer, request, compact_standalone=True) for method in methods]
        actual = [future.result() for future in futures]
    for desired, observed, mode in zip(expected, actual, ("original", "canonical")):
        assert observed["selection"]["evidence_tokenization"] == mode
        assert observed["teacher"]["evidence_tokenization"] == mode
        assert observed["teacher"]["tokens"] == desired["teacher"]["tokens"]
        assert observed["generated_token_ids"] == desired["generated_token_ids"]
    assert expected[0]["teacher"]["prompt_tokens"] == expected[1]["teacher"]["prompt_tokens"] + 2


def test_evidence_context_restores_state_after_failure_and_rejects_invalid_mode():
    assert _CURRENT.get() is None
    with pytest.raises(RuntimeError, match="failure"):
        with selected_evidence_mode(object(), "canonical", sink_tokens=4):
            assert _CURRENT.get() is not None
            raise RuntimeError("failure")
    assert _CURRENT.get() is None
    with pytest.raises(ValueError, match="Unknown selected evidence"):
        with selected_evidence_mode(None, "unexpected", sink_tokens=4):
            pass
