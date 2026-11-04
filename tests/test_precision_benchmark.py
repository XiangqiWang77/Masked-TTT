"""A precision comparison uses the same SD state and releases both artifacts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments.precision_cartridge.benchmark_runner import precision_controls, precision_runner, preflight
from experiments.precision_cartridge.benchmark_report import summarize, validate_result
from experiments.precision_cartridge.materialization import float32_cartridge
from experiments.anchored_short_verifier import paired_controls
from experiments.anchored_short_verifier.paired_controls import original_question_controls
from masked_ttt import MaskedTTT, Request
from masked_ttt.linked_selection import LinkedFrozenSelector
from masked_ttt.kv_only import kv_only_cartridge
from test_materialization_equivalence import gpt_gqa_engine
from test_public_linked_selection import _source
from test_standalone import make_engine
import masked_ttt as runtime


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_bf16_and_float32_cartridges_share_training_and_release_state(architecture, tmp_path, monkeypatch):
    monkeypatch.setenv("MASKED_TTT_FIXED_CHAT_DATE", "2026-11-03")
    document, question, _, _ = _source(tmp_path)
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.model.to(dtype=torch.bfloat16)
    engine.config = replace(engine.config, keep_fraction=.1, student_rollout_tokens=32,
        self_question_count=0, guard_enabled=False, selected_transfer_weight=0., gain_head_mode="kv",
        value_basis_source="document", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    engine.eos_ids = {-1}
    selector = LinkedFrozenSelector(tmp_path)
    selector.bind_case("case")
    with precision_controls():
        result = MaskedTTT(engine, selector).answer(Request(
            document=document, question=question, max_new_tokens=4))
    diagnostic = result["standalone_cartridge"]["precision_paired_diagnostic"]
    assert diagnostic["passed"], diagnostic.get("error")
    assert diagnostic["precision_tokens"] == diagnostic["precision_matched_adapted_tokens"]
    assert diagnostic["precision_kv_only_materialized"] and diagnostic["precision_adapter_bytes"] == 0
    assert diagnostic["selected_source_bytes_before_decode"] == 0
    assert diagnostic["training_bytes_before_decode"] == 0
    assert diagnostic["same_selected_kv"] and diagnostic["same_question_suffix"]
    assert diagnostic["same_precision_backend"] and not diagnostic["reference_answers_used"]
    bf16 = result["standalone_cartridge"]["memory"]
    assert bf16["kv_only_materialized"] and bf16["cartridge_adapter_bytes"] == 0
    assert bf16["additional_source_kv_bytes_before_decode"] == 0
    assert bf16["additional_training_adapter_bytes_before_decode"] == 0
    assert result["training"]["objective"] == "on-policy KL(teacher || student)"
    assert result["training"]["selected_transfer_weight"] == 0
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    model = "qwen" if architecture == "qwen3" else "gpt"
    model_id = "Qwen/Qwen3-4B-Instruct-2507" if model == "qwen" else "openai/gpt-oss-20b"
    result.update(key="case", model=model_id, task="qa1", keep_fraction=.1)
    result["self_judge"]["native_prediction"] = "blue"
    validate_result(result, "case", model, "qa1")


def test_precision_runner_restores_all_hooks_on_failure():
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda value: value,
        validate=lambda _result, _entry: None, preflight=lambda _engine: None)
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.preflight, runtime.FrozenJSONSelector, paired_controls.original_question_controls)
    with pytest.raises(RuntimeError, match="intentional"):
        with precision_runner(implementation, runtime, paired_controls, root="candidate",
                              linked_selector=LinkedFrozenSelector):
            raise RuntimeError("intentional")
    assert (implementation.ROOT, implementation.validate_manifest, implementation.validate,
            implementation.preflight, runtime.FrozenJSONSelector, paired_controls.original_question_controls) == old


def test_precision_preflight_runs_inside_the_patched_runner_without_recursion():
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda value: value,
        validate=lambda _result, _entry: None, preflight=lambda _engine: None)
    engine = make_engine()
    engine.model.to(dtype=torch.bfloat16)
    with precision_runner(implementation, runtime, paired_controls, root="candidate",
                          linked_selector=LinkedFrozenSelector):
        assert implementation.preflight(engine)["passed"]


def test_precision_runner_composes_with_frozen_selector_and_real_request(tmp_path):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine()
    engine.model.to(dtype=torch.bfloat16)
    engine.config = replace(engine.config, keep_fraction=.1, student_rollout_tokens=32,
        self_question_count=0, guard_enabled=False, selected_transfer_weight=0., gain_head_mode="kv",
        value_basis_source="document", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda value: value,
        validate=lambda _result, _entry: None, preflight=lambda _engine: None)
    with precision_runner(implementation, runtime, paired_controls, root="candidate",
                          linked_selector=LinkedFrozenSelector):
        selector = runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with paired_controls.original_question_controls():
            result = MaskedTTT(engine, selector).answer(Request(
                document=document, question=question, max_new_tokens=3))
        implementation.validate(result, {"key": "case"})
    assert result["standalone_cartridge"]["precision_paired_diagnostic"]["passed"]


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_precision_preflight_uses_both_paths_and_restores_config(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.model.to(dtype=torch.bfloat16)
    previous = engine.config
    assert preflight(engine)["passed"]
    assert engine.config is previous


def test_precision_summary_preserves_matched_scores_and_sizes():
    rows = [dict(case="babilong-8-qa3-1", whole=0, teacher=1, bf16_zero=0, bf16=0,
                 fp32_zero=0, fp32=1, bf16_kv_percent=15., fp32_kv_percent=20.),
            dict(case="babilong-16-qa3-1", whole=1, teacher=0, bf16_zero=1, bf16=1,
                 fp32_zero=1, fp32=0, bf16_kv_percent=13., fp32_kv_percent=19.)]
    result = summarize(rows)
    assert result["n"] == 2 and result["source_questions"] == 1
    assert result["fp32_vs_bf16"] == dict(new_correct=1, new_wrong=1, delta_pp=0.)
    assert result["bf16_kv_percent"] == 14. and result["fp32_kv_percent"] == 19.5


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_precision_pilot_accepts_a_true_zero_update_without_inventing_a_gain(tmp_path, monkeypatch, architecture):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.model.to(dtype=torch.bfloat16)
    engine.config = replace(engine.config, keep_fraction=.1, student_rollout_tokens=32,
        self_question_count=0, guard_enabled=False, selected_transfer_weight=0., gain_head_mode="kv",
        value_basis_source="document", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    selector = LinkedFrozenSelector(tmp_path)
    selector.bind_case("case")
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, *args, **kwargs: None)
    method = MaskedTTT(engine, selector)
    request = Request(document=document, question=question, max_new_tokens=3)
    with kv_only_cartridge(), original_question_controls():
        baseline = method.answer(request)
    with precision_controls():
        result = method.answer(request)
    diag = result["standalone_cartridge"]["precision_paired_diagnostic"]
    assert diag["passed"], diag.get("error")
    assert result["generated_token_ids"] == baseline["generated_token_ids"]
    assert result["teacher"]["tokens"] == baseline["teacher"]["tokens"]
    assert diag["precision_tokens"] == diag["precision_zero"]["tokens"]
    assert diag["precision_kv_only_materialized"] and diag["precision_adapter_bytes"] == 0


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_float32_cartridge_can_serve_without_a_bf16_artifact_or_training_state(architecture, tmp_path):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.model.to(dtype=torch.bfloat16)
    engine.config = replace(engine.config, keep_fraction=.1, self_question_count=0, guard_enabled=False,
        gain_head_mode="kv", selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    selector = LinkedFrozenSelector(tmp_path)
    selector.bind_case("case")
    with kv_only_cartridge(), float32_cartridge(matched_controls=True):
        result = MaskedTTT(engine, selector).answer(Request(
            document=document, question=question, max_new_tokens=3))
    memory = result["standalone_cartridge"]["memory"]
    assert result["output_source"] == "standalone_cartridge"
    assert memory["experimental_materialization"] == "float32_value_cache"
    assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0
    assert memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0
