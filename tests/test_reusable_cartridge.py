from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from masked_ttt import MaskedTTT, QueryCartridge, Request
from masked_ttt.memory_accounting import module_tensors
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_materialization_equivalence import gpt_gqa_engine
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def setup(architecture, steps=2):
    engine = (make_engine() if architecture == "qwen3" else
              gpt_engine() if architecture == "gpt_oss" else gpt_gqa_engine())
    engine.config = replace(engine.config, steps=steps, self_question_count=0, guard_enabled=False,
        selected_transfer_weight=0., value_basis_source="document", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=3)
    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks)-2], dict(backend="unit"))
    return engine, MaskedTTT(engine, Selector()), request


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss", "gpt_oss_gqa"])
@pytest.mark.parametrize("steps", [0, 2])
def test_cartridge_reanswers_from_independent_immutable_selected_kv(architecture, steps):
    engine, method, request = setup(architecture, steps)
    cartridge, report = method.prepare_cartridge(request)
    assert isinstance(cartridge, QueryCartridge)
    memory = report["standalone_cartridge"]["memory"]
    assert report["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    assert memory["kv_only_materialized"] and memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0
    assert not module_tensors(cartridge._reader.memory)
    assert cartridge.document_kv_bytes == memory["materialized_selected_kv_bytes"]
    one, two = cartridge.answer(), cartridge.answer()
    assert one["generated_token_ids"] == report["generated_token_ids"] == two["generated_token_ids"]
    assert one["prediction"] == report["prediction"] == two["prediction"]
    assert one["original_question_only"] and not one["teacher_answer_in_final_prompt"]
    # Each generated suffix belongs to a private branch; repeated calls and
    # concurrent callers leave the retained source positions unchanged.
    with ThreadPoolExecutor(max_workers=2) as pool:
        repeated = list(pool.map(lambda _: cartridge.answer(), range(2)))
    assert all(item["generated_token_ids"] == one["generated_token_ids"] for item in repeated)
    assert all(not p.requires_grad and p.grad is None for p in engine.model.parameters())
    branch, suffix = weakref.ref(cartridge._branch), weakref.ref(cartridge._suffix)
    cartridge.close()
    gc.collect()
    assert branch() is None and suffix() is None
    with pytest.raises(RuntimeError, match="closed"):
        cartridge.answer()
    with pytest.raises(RuntimeError, match="closed"):
        _ = cartridge.document_kv_bytes


@pytest.mark.parametrize("overrides", [
    dict(release_full_before_cartridge=False),
    dict(release_selected_source_before_decode=False),
    dict(standalone_cartridge_enabled=False, release_full_before_cartridge=False),
])
def test_cartridge_requires_sources_to_be_released_before_selection(overrides):
    engine, method, request = setup("qwen3")
    engine.config = replace(engine.config, **overrides)
    class FailSelector:
        def prepare(self, *_):
            pytest.fail("Invalid release configuration must fail before selecting")
    method.selector = FailSelector()
    with pytest.raises(ValueError, match="release of both source caches"):
        method.prepare_cartridge(request)


def test_cartridge_rejects_changed_cache_and_invalid_request_budget():
    _, method, request = setup("qwen3")
    cartridge, _ = method.prepare_cartridge(request)
    for value in (0, -1, True, float("nan")):
        with pytest.raises(ValueError, match="time_budget_seconds"):
            cartridge.answer(time_budget_seconds=value)
    with pytest.raises(ValueError, match="max_new_tokens"):
        cartridge.answer(max_new_tokens=True)
    with torch.no_grad():
        cartridge._branch.layers[0].keys[..., 0, 0].add_(1)
    with pytest.raises(RuntimeError, match="KV was modified"):
        cartridge.answer()
    cartridge.close()


def test_cartridge_detects_changed_backbone():
    engine, method, request = setup("qwen3")
    cartridge, _ = method.prepare_cartridge(request)
    with torch.no_grad():
        next(engine.model.parameters()).add_(1)
    with pytest.raises(AssertionError, match="backbone weight"):
        cartridge.answer()
    cartridge.close()


