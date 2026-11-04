from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from experiments.value_lr_physical.canonical_pair import run_pair, validate_pair
from masked_ttt import Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


class BoundaryTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **_kwargs):
        return "USER:\n" + messages[0]["content"] + "\nASSISTANT:\n"

    def encode(self, text, **_kwargs):
        return self(text)["input_ids"]

    def __call__(self, text, **_kwargs):
        ids, offsets, index = [], [], 0
        while index < len(text):
            if text.startswith("\n\nE", index):
                ids.append(300); offsets.append((index, index + 3)); index += 3
            else:
                ids.append(ord(text[index])); offsets.append((index, index + 1)); index += 1
        return dict(input_ids=ids, offset_mapping=offsets)

    def decode(self, ids, **_kwargs):
        return "".join("\n\nE" if int(value) == 300 else chr(int(value)) for value in ids)


class Selector:
    def prepare(self, backbone, request):
        layout = prepare_layout(backbone, request)
        assert len(layout.chunks) == 4
        selected = [0, 2]
        metadata = dict(backend="unit", selected_original_chunks=selected, selected_chunks=selected,
                        total_chunks=len(layout.chunks),
                        selected_tokens=sum(len(layout.chunks[index]) for index in selected))
        return PreparedRequest(request, layout, selected, metadata)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_pair_preserves_source_and_release_with_a_real_token_boundary_change(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    if architecture == "qwen3":
        engine.tokenizer = BoundaryTokenizer()
    else:
        class HarmonyBoundaryTokenizer(BoundaryTokenizer):
            def apply_chat_template(self, messages, **_kwargs):
                return "<|start|>user<|message|>" + messages[0]["content"] + "<|end|><|start|>assistant"
        from masked_ttt.architectures import FinalChannelTokenizer
        engine.tokenizer = FinalChannelTokenizer(HarmonyBoundaryTokenizer())
    config = replace(engine.config, chunk_tokens=32, steps=2, self_question_count=0,
        guard_enabled=True, selected_transfer_weight=0., value_basis_source="document",
        standalone_answer_format="original", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request = Request(document="A" * 32 + "B" * 32 + "E" * 32 + "F" * 32,
                      question="Which block is E?", max_new_tokens=3)
    rng, original_config = torch.get_rng_state().clone(), engine.config
    result = run_pair(engine, Selector(), request, config)
    assert torch.equal(rng, torch.get_rng_state()) and engine.config is original_config
    assert result["same_loaded_model"] and result["same_device"] and not result["same_gpu"]
    assert result["independently_prefilled_caches"] and not result["reference_answers_used"]
    assert result["changed_settings"] == ["selected_evidence_tokenization"]
    left, right = result["left"], result["right"]
    assert left["teacher"]["prompt_tokens"] == right["teacher"]["prompt_tokens"] + 2
    assert left["self_judge"]["native_prediction"] == right["self_judge"]["native_prediction"]
    for arm in (left, right):
        assert arm["training"]["objective"] == "on-policy KL(teacher || student)"
        assert arm["standalone_cartridge"]["memory"]["kv_only_materialized"]
    broken = deepcopy(right)
    broken["selection"]["selected_chunks"] = [1]
    with pytest.raises(ValueError, match="source chunks"):
        validate_pair(left, broken)
    broken = deepcopy(right)
    broken["standalone_cartridge"]["memory"]["additional_source_kv_bytes_before_decode"] = 1
    with pytest.raises(ValueError, match="cartridge contract"):
        validate_pair(left, broken)


def test_requires_native_control_and_one_shared_deadline():
    engine = make_engine()
    config = replace(engine.config, guard_enabled=False, selected_transfer_weight=0.,
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    with pytest.raises(ValueError, match="native full-context"):
        run_pair(engine, Selector(), None, config)
    before = torch.get_rng_state().clone()
    with pytest.raises(TimeoutError, match="paired request budget"):
        run_pair(engine, Selector(), None, replace(config, guard_enabled=True), time_budget_seconds=0)
    assert torch.equal(before, torch.get_rng_state())
