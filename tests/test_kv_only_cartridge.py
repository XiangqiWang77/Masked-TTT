from dataclasses import replace

import pytest
import torch

from experiments.kv_only_cartridge.compact import kv_only_cartridge
from masked_ttt.kv_only import kv_only_cartridge as installed_kv_only_cartridge
from masked_ttt import MaskedTTT, Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def test_installed_and_experimental_entrypoints_share_the_same_runtime():
    assert installed_kv_only_cartridge is kv_only_cartridge


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
@pytest.mark.parametrize("steps", [0, 2])
def test_kv_only_preserves_tokens_and_frees_reader_tensors(architecture, steps, monkeypatch):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    # A no-op optimizer step exercises the native read path without altering
    # the trained/masked path or requiring an invalid zero learning rate.
    engine.config = replace(
        engine.config,
        steps=2,
        gain_learning_rate=.01,
        value_learning_rate=.0001,
        self_question_count=0,
        guard_enabled=False,
        selected_transfer_weight=0.0,
        value_basis_source="document",
        release_full_before_cartridge=True,
        release_selected_source_before_decode=True,
    )
    request = Request(
        document="The old note is red. The current note is blue. " * 6,
        question="Which color is current?",
        max_new_tokens=4,
    )

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    method = MaskedTTT(engine, Selector())
    logits = []
    handle = engine.model.lm_head.register_forward_hook(
        lambda module, args, output: logits.append(output.detach().clone())
    )
    with monkeypatch.context() as patch:
        if steps == 0:
            patch.setattr(torch.optim.Adam, "step", lambda self, *args, **kwargs: None)
        baseline = method.answer(request)
        baseline_logits, logits = logits, []
        with kv_only_cartridge():
            result = method.answer(request)
    handle.remove()
    assert result["generated_token_ids"] == baseline["generated_token_ids"]
    assert len(baseline_logits) == len(logits) > 0
    assert all(torch.equal(left, right) for left, right in zip(baseline_logits, logits))
    assert result["teacher"]["tokens"] == baseline["teacher"]["tokens"]
    assert result["training"]["objective"] == baseline["training"]["objective"]
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0

    source = baseline["standalone_cartridge"]["memory"]
    actual = result["standalone_cartridge"]["memory"]
    assert actual["kv_only_materialized"]
    assert actual["kv_only_native_mode"] == (steps == 0)
    assert actual["kv_only_adapter_bytes_removed"] == source["cartridge_adapter_bytes"] > 0
    assert actual["cartridge_adapter_bytes"] == 0
    assert actual["standalone_cartridge_bytes"] == actual["materialized_selected_kv_bytes"]
    assert actual["standalone_cartridge_bytes"] < source["standalone_cartridge_bytes"]
    assert actual["additional_source_kv_bytes_before_decode"] == 0
    assert actual["additional_training_adapter_bytes_before_decode"] == 0


def test_hook_is_restored_after_error():
    import masked_ttt.pipeline as pipeline

    original = pipeline.answer_standalone
    with pytest.raises(RuntimeError, match="intentional"):
        with kv_only_cartridge():
            raise RuntimeError("intentional")
    assert pipeline.answer_standalone is original


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss", "gpt_oss_gqa"])
@pytest.mark.parametrize("steps", [0, 2])
def test_request_local_compact_path_matches_existing_kv_only(architecture, steps):
    import masked_ttt.pipeline as pipeline
    from test_materialization_equivalence import gpt_gqa_engine

    engine = (make_engine() if architecture == "qwen3" else
              gpt_engine() if architecture == "gpt_oss" else gpt_gqa_engine())
    engine.config = replace(engine.config, steps=steps, self_question_count=0,
        guard_enabled=False, selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    method = MaskedTTT(engine, Selector())
    with kv_only_cartridge():
        legacy = method.answer(request)
    answer_hook, decode_hook = pipeline.answer_standalone, pipeline.decode_standalone
    direct = method.answer(request, compact_standalone=True)
    assert pipeline.answer_standalone is answer_hook and pipeline.decode_standalone is decode_hook
    assert direct["generated_token_ids"] == legacy["generated_token_ids"]
    assert direct["teacher"]["tokens"] == legacy["teacher"]["tokens"]
    assert direct["training"]["objective"] == legacy["training"]["objective"]
    assert direct["training"]["accepted_steps"] == legacy["training"]["accepted_steps"]
    assert direct["backbone_unchanged"] and direct["original_document_KV_unchanged"]
    assert direct["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    memory = direct["standalone_cartridge"]["memory"]
    assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0
    assert memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0
    assert memory["standalone_cartridge_bytes"] == legacy["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]


@pytest.mark.parametrize("overrides", [
    dict(release_full_before_cartridge=False),
    dict(release_selected_source_before_decode=False),
    dict(standalone_cartridge_enabled=False, release_full_before_cartridge=False),
])
def test_request_local_compaction_rejects_source_retention_before_selection(overrides):
    engine = make_engine()
    engine.config = replace(engine.config, **overrides)
    class Selector:
        def prepare(self, *_args):
            pytest.fail("Invalid compact settings must be rejected before selection")
    method = MaskedTTT(engine, Selector())
    request = Request(document="A short document.", question="What is it?", max_new_tokens=3)
    with pytest.raises(ValueError, match="release of both source caches"):
        method.answer(request, compact_standalone=True)
    with pytest.raises(ValueError, match="must be boolean"):
        method.answer(request, compact_standalone="true")


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_request_local_compaction_keeps_parallel_requests_isolated(architecture):
    from concurrent.futures import ThreadPoolExecutor
    from test_materialization_equivalence import gpt_gqa_engine

    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.config = replace(engine.config, self_question_count=0, guard_enabled=False,
        selected_transfer_weight=0., release_full_before_cartridge=True,
        release_selected_source_before_decode=True)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    method = MaskedTTT(engine, Selector())
    requests = [Request(document="The old note is red. The current note is blue. " * 6,
                        question=question, max_new_tokens=3)
                for question in ("Which color is current?", "Which color is old?")]
    expected = [method.answer(request, compact_standalone=True) for request in requests]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(method.answer, request, compact_standalone=True) for request in requests]
        observed = [future.result() for future in futures]
    for baseline, result in zip(expected, observed):
        assert result["generated_token_ids"] == baseline["generated_token_ids"]
        assert result["teacher"]["tokens"] == baseline["teacher"]["tokens"]
        assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
        assert result["standalone_cartridge"]["memory"]["cartridge_adapter_bytes"] == 0


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_loaded_backbone_probe_preserves_configuration(architecture):
    from experiments.kv_only_cartridge.real_model_probe import validate_loaded_backbone

    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    original = engine.config
    result = validate_loaded_backbone(engine)
    assert result["passed"] and result["same_standalone_logits"]
    assert result["long_attention_positions"] == 32 and result["long_attention_gradients_match"]
    assert result["cartridge_bytes_after"] < result["cartridge_bytes_before"]
    assert engine.config is original


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_kv_only_keeps_same_prompt_pairing_controls(architecture):
    from experiments.anchored_short_verifier.paired_controls import original_question_controls

    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    engine.config = replace(
        engine.config,
        self_question_count=0,
        guard_enabled=False,
        selected_transfer_weight=0.0,
        release_full_before_cartridge=True,
        release_selected_source_before_decode=True,
    )
    request = Request(
        document="The old note is red. The current note is blue. " * 6,
        question="Which color is current?",
        max_new_tokens=3,
    )

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    with original_question_controls(), kv_only_cartridge():
        result = MaskedTTT(engine, Selector()).answer(request)
    standalone = result["standalone_cartridge"]
    assert standalone["original_question_controls"]["adapted"]["tokens"] == result["generated_token_ids"]
    assert standalone["memory"]["kv_only_materialized"]
    assert standalone["memory"]["cartridge_adapter_bytes"] == 0


def test_optional_driver_restores_rng_and_configuration(tmp_path):
    import json
    from types import SimpleNamespace

    from experiments.kv_only_cartridge.driver import run_if_enabled

    original = object()
    engine = SimpleNamespace(config=original)
    assert run_if_enabled(engine, "qwen", {}, tmp_path) == "not_enabled"
    (tmp_path / "ready.json").write_text(json.dumps({"allow_single_loaded_model_probe": True}))
    before = torch.get_rng_state()

    def probe(model):
        model.config = object()
        torch.manual_seed(123456)
        return {"passed": True, "same_standalone_logits": True}

    assert run_if_enabled(engine, "qwen", {"gpu": "test"}, tmp_path, probe=probe) == "passed"
    assert engine.config is original and torch.equal(torch.get_rng_state(), before)
    stored = json.loads((tmp_path / "qwen.json").read_text())
    assert stored["same_standalone_logits"] and stored["identity"]["gpu"] == "test"
    assert run_if_enabled(engine, "qwen", {}, tmp_path, probe=probe) == "already_passed"
    (tmp_path / "ready.json").write_text(json.dumps(
        {"allow_single_loaded_model_probe": True, "probe_version": 2}))
    assert run_if_enabled(engine, "qwen", {}, tmp_path, probe=probe) == "passed"
    updated = json.loads((tmp_path / "qwen.json").read_text())
    assert updated["probe_version"] == 2 and updated["attempts"] == 1
    assert json.loads((tmp_path / "qwen.v1.json").read_text())["probe_version"] == 1
    assert engine.config is original and torch.equal(torch.get_rng_state(), before)


def test_optional_driver_saves_failure_without_disturbing_model(tmp_path):
    import json
    from types import SimpleNamespace

    from experiments.kv_only_cartridge.driver import run_if_enabled

    (tmp_path / "ready.json").write_text(json.dumps({"allow_single_loaded_model_probe": True}))
    original = object()
    engine = SimpleNamespace(config=original)
    before = torch.get_rng_state()

    def fail(model):
        model.config = object()
        torch.manual_seed(123456)
        raise ValueError("probe failed")

    assert run_if_enabled(engine, "gpt", {}, tmp_path, probe=fail) == "failed"
    assert engine.config is original and torch.equal(torch.get_rng_state(), before)
    assert json.loads((tmp_path / "gpt.json").read_text())["error"] == "ValueError"
