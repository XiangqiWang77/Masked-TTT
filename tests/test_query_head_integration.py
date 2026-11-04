"""Query-head gains retain the original full-SD and standalone-KV contracts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from masked_ttt import MaskedTTT, Request
from masked_ttt.cache import fork_prefix
from masked_ttt.batched_sd import batched_teacher_logits
from masked_ttt.checkpointing import layer_checkpointed_logits
from masked_ttt.kv_only import kv_only_cartridge
from masked_ttt.memory import KVReadMemory
from masked_ttt.model import Budget
from masked_ttt.reader import MemoryReader, matched_incremental_logits
from masked_ttt.request import prepare_layout
from masked_ttt.runtime import private_forward, private_module_tree
from masked_ttt.selection import PreparedRequest
from masked_ttt.standalone import build_standalone
from masked_ttt.teacher import build_teacher, prefill_document
from test_materialization_equivalence import gpt_gqa_engine
from test_standalone import make_engine


def setup(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.config = replace(engine.config, self_question_count=0, guard_enabled=False,
                            selected_transfer_weight=0., value_basis_source="document")
    engine.budget = Budget(None)
    request = Request(document="The old note is red. The current note is blue. " * 5,
                      question="Which color is current?", max_new_tokens=4)
    layout = prepare_layout(engine, request)
    selected = [1, len(layout.chunks) - 2]
    full = prefill_document(engine, engine.model, layout.ids, layout.initial,
                            [chunk[-1] + 1 for chunk in layout.chunks])
    teacher_reader = SimpleNamespace(model=private_module_tree(engine.model.model),
                                     lm_head=engine.model.lm_head)
    teacher = build_teacher(engine, teacher_reader, request, layout, selected)
    return engine, request, layout, selected, full, teacher


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_query_head_materialization_matches_dynamic_values_and_gradients(architecture):
    engine, _, layout, selected, full, teacher = setup(architecture)
    memory = KVReadMemory(engine.model.config, layout.chunks, selected, layout.document_end,
                          rank=2, gain_head_mode="query", sink_tokens=engine.config.sink_tokens)
    assert memory.gain.shape[1] == engine.model.config.num_attention_heads > memory.heads
    memory.prepare_value_basis(full, memory.adapted_layers)
    memory.value_enabled = True
    torch.manual_seed(71)
    with torch.no_grad():
        for layer in memory.adapted_layers:
            memory.gain[layer, :, selected[0]] = torch.tensor([.17, -.08])
            memory.gain[layer, :, selected[1]] = torch.tensor([-.11, .14])
            memory.value_coefficients[layer].normal_(std=.025)
    reader, branch, suffix, length, _ = build_standalone(engine, layout, selected, teacher, memory)
    target = reader.memory
    for parameter in target.parameters():
        parameter.requires_grad_(True)
    source = teacher.document_cache
    stamps = [(tensor, tensor._version, tensor.clone()) for layer in source.layers
              for tensor in (layer.keys, layer.values) if tensor is not None]
    assert all(branch.layers[i].keys.shape[1] == memory.heads for i in memory.adapted_layers)

    def compute(compiled):
        fresh = reader.compiled_branch(source) if compiled else fork_prefix(source, length)
        return private_forward(reader.reader, suffix, length, fresh).last_hidden_state

    dynamic, materialized = compute(False), compute(True)
    torch.testing.assert_close(dynamic, materialized, atol=2e-6, rtol=2e-5)
    torch.manual_seed(73)
    probe = torch.randn_like(dynamic)
    gradients = [torch.autograd.grad((output * probe).sum(),
                 (target.gain, target.value_coefficients)) for output in (dynamic, materialized)]
    for first, second in zip(*gradients):
        assert first.isfinite().all() and first.abs().sum() > 0
        torch.testing.assert_close(first, second, atol=2e-6, rtol=3e-5)
    gain_gradient = gradients[0][0]
    assert not torch.equal(gain_gradient[:, 0], gain_gradient[:, 1])
    assert not gain_gradient[:, :, ~target.selected_chunk_mask].any()
    for tensor, version, original in stamps:
        assert tensor._version == version
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_shared_query_head_gain_matches_original_shared_kv_gain(architecture):
    engine, _, layout, selected, full, _ = setup(architecture)
    shared = KVReadMemory(engine.model.config, layout.chunks, selected, layout.document_end,
                          rank=2, gain_head_mode="kv", sink_tokens=engine.config.sink_tokens)
    individual = KVReadMemory(engine.model.config, layout.chunks, selected, layout.document_end,
                              rank=2, gain_head_mode="query", sink_tokens=engine.config.sink_tokens)
    shared.prepare_value_basis(full, shared.adapted_layers)
    individual.value_basis.copy_(shared.value_basis)
    individual.value_layers.copy_(shared.value_layers)
    individual.active_value_layers = set(shared.active_value_layers)
    shared.value_enabled = individual.value_enabled = True
    torch.manual_seed(75)
    with torch.no_grad():
        shared.gain.normal_(std=.1)
        shared.constrain()
        individual.gain.copy_(shared.gain.repeat_interleave(
            individual.query_heads // individual.heads, dim=1))
        shared.value_coefficients.normal_(std=.01)
        individual.value_coefficients.copy_(shared.value_coefficients)
    suffix = torch.tensor(layout.ids[layout.document_end:])
    results = []
    for state in (shared, individual):
        reader = MemoryReader(engine, state)
        with torch.no_grad():
            results.append(private_forward(reader.reader, suffix, layout.document_end,
                                           reader.compiled_branch(full)).last_hidden_state)
    torch.testing.assert_close(*results, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_query_head_causal_sd_backpropagation_matches_incremental_and_checkpointed(architecture):
    engine, _, layout, selected, full, _ = setup(architecture)
    memory = KVReadMemory(engine.model.config, layout.chunks, selected, layout.document_end,
                          rank=2, gain_head_mode="query", sink_tokens=engine.config.sink_tokens)
    memory.prepare_value_basis(full, memory.adapted_layers)
    memory.value_enabled = True
    torch.manual_seed(77)
    with torch.no_grad():
        memory.gain.normal_(std=.1)
        memory.constrain()
        memory.value_coefficients.normal_(std=.01)
    reader = MemoryReader(engine, memory)
    suffix = torch.tensor(layout.ids[layout.document_end:])
    targets = torch.tensor([41, 42, 43])
    logits = [batched_teacher_logits(engine, reader, full, suffix, targets),
              matched_incremental_logits(engine, reader, full, suffix, targets),
              layer_checkpointed_logits(engine, reader, full, suffix, targets)]
    probe = torch.randn_like(logits[0])
    gradients = [torch.autograd.grad((output * probe).sum(),
                                     (memory.gain, memory.value_coefficients)) for output in logits]
    for output in logits[1:]:
        torch.testing.assert_close(output, logits[0], atol=2e-6, rtol=2e-5)
    for result in gradients[1:]:
        for actual, expected in zip(result, gradients[0]):
            assert actual.isfinite().all() and actual.abs().sum() > 0
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_query_head_mode_updates_both_paths_and_decodes_a_physically_independent_cartridge(architecture):
    engine, request, _, _, _, _ = setup(architecture)
    engine.config = replace(engine.config, gain_head_mode="query", steps=2,
                            release_full_before_cartridge=True,
                            release_selected_source_before_decode=True)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    with kv_only_cartridge():
        result = MaskedTTT(engine, Selector()).answer(request)
    train = result["training"]
    assert train["gain_head_mode"] == "query"
    assert train["objective"] == "on-policy KL(teacher || student)"
    assert train["selected_transfer_weight"] == 0 and train["steps"]
    assert train["steps"][0]["gain_gradient_norm"] > 0
    assert train["steps"][0]["value_gradient_norm"] > 0
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    assert result["prediction"] == result["standalone_cartridge"]["prediction"]
    physical = result["standalone_cartridge"]["memory"]
    assert physical["kv_only_materialized"]
    assert physical["additional_source_kv_bytes_before_decode"] == 0
    assert physical["additional_training_adapter_bytes_before_decode"] == 0
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
