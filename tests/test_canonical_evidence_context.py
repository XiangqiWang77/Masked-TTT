from dataclasses import replace

import pytest
import torch

from experiments.value_lr_physical.canonical_context import canonical_selected_evidence
from masked_ttt import MaskedTTT, Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_opt_in_canonical_path_matches_native_when_tokenization_is_already_canonical(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    raw_tokenizer = engine.tokenizer if architecture == "qwen3" else engine.tokenizer.tokenizer
    raw_tokenizer.decode = lambda ids, **_kwargs: "".join(chr(int(value)) for value in ids)
    engine.config = replace(engine.config, self_question_count=1, self_question_max_tokens=8,
        guard_enabled=False, selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=3)
    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks)-2], dict(backend="unit"))
    method = MaskedTTT(engine, Selector())
    original = method.answer(request, compact_standalone=True)
    with canonical_selected_evidence(engine.tokenizer, sink_tokens=engine.config.sink_tokens):
        canonical = method.answer(request, compact_standalone=True)
    assert canonical["teacher"]["tokens"] == original["teacher"]["tokens"]
    assert canonical["generated_token_ids"] == original["generated_token_ids"]
    assert canonical["self_study"]["checks"] == original["self_study"]["checks"]
    assert len(canonical["training"]["steps"]) == len(original["training"]["steps"])
    for a, b in zip(original["training"]["steps"], canonical["training"]["steps"]):
        assert a["student_rollout_tokens"] == b["student_rollout_tokens"]
        assert a["teacher_KL_before"] == pytest.approx(b["teacher_KL_before"], abs=1e-7)
        assert a["gain_gradient_norm"] == pytest.approx(b["gain_gradient_norm"], abs=1e-7)
        assert a["value_gradient_norm"] == pytest.approx(b["value_gradient_norm"], abs=1e-7)
    assert canonical["backbone_unchanged"] and canonical["original_document_KV_unchanged"]
    assert canonical["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    memory = canonical["standalone_cartridge"]["memory"]
    assert memory["kv_only_materialized"] and memory["additional_source_kv_bytes_before_decode"] == 0


def test_context_restores_all_planning_hooks_after_error():
    from masked_ttt import self_study_runtime, shared_selected, standalone, teacher
    modules = (teacher, self_study_runtime, shared_selected, standalone)
    originals = [module.selected_read_plan for module in modules]
    old_standalone, old_shared = standalone.selected_chunk_positions, shared_selected.selected_chunk_positions
    with pytest.raises(RuntimeError, match="deliberate"):
        with canonical_selected_evidence(None, sink_tokens=4):
            assert all(module.selected_read_plan is not original for module, original in zip(modules, originals))
            raise RuntimeError("deliberate")
    assert all(module.selected_read_plan is original for module, original in zip(modules, originals))
    assert standalone.selected_chunk_positions is old_standalone and shared_selected.selected_chunk_positions is old_shared


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_canonical_teacher_and_materialized_cartridge_share_merged_boundary_token(architecture):
    class BoundaryTokenizer:
        eos_token_id = 2
        def apply_chat_template(self, messages, **_kwargs):
            return "USER:\n" + messages[0]["content"] + "\nASSISTANT:\n"
        def encode(self, text, **_kwargs):
            return self(text)["input_ids"]
        def __call__(self, text, **_kwargs):
            ids, offsets, i = [], [], 0
            while i < len(text):
                if text.startswith("\n\nE", i):
                    ids.append(300); offsets.append((i, i+3)); i += 3
                else:
                    ids.append(ord(text[i])); offsets.append((i, i+1)); i += 1
            return dict(input_ids=ids, offset_mapping=offsets)
        def decode(self, ids, **_kwargs):
            return "".join("\n\nE" if int(value) == 300 else chr(int(value)) for value in ids)
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    if architecture == "qwen3":
        engine.tokenizer = BoundaryTokenizer()
    else:
        class HarmonyBoundaryTokenizer(BoundaryTokenizer):
            def apply_chat_template(self, messages, **_kwargs):
                return "<|start|>user<|message|>" + messages[0]["content"] + "<|end|><|start|>assistant"
        from masked_ttt.architectures import FinalChannelTokenizer
        engine.tokenizer = FinalChannelTokenizer(HarmonyBoundaryTokenizer())
    engine.config = replace(engine.config, chunk_tokens=32, self_question_count=0, guard_enabled=False,
        selected_transfer_weight=0., value_basis_source="document", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request = Request(document="A"*32 + "B"*32 + "E"*32 + "F"*32,
                      question="Which block is E?", max_new_tokens=3)
    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            assert len(layout.chunks) == 4
            return PreparedRequest(item, layout, [0, 2], dict(backend="unit"))
    method = MaskedTTT(engine, Selector())
    raw = method.answer(request, compact_standalone=True)
    with canonical_selected_evidence(engine.tokenizer, sink_tokens=engine.config.sink_tokens):
        canonical = method.answer(request, compact_standalone=True)
    assert raw["teacher"]["prompt_tokens"] == canonical["teacher"]["prompt_tokens"] + 2
    assert canonical["training"]["objective"] == "on-policy KL(teacher || student)"
    assert canonical["backbone_unchanged"] and canonical["original_document_KV_unchanged"]
    assert canonical["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    assert canonical["standalone_cartridge"]["memory"]["additional_source_kv_bytes_before_decode"] == 0
