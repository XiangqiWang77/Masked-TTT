"""Development-only LR selection and the standalone original-SD runtime."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

import experiments.anchored_short_verifier.paired_controls as paired_controls
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.value_lr_physical.protocol import development_choice, holdout_manifest, name_for_lr
from experiments.value_lr_physical.report import validate_result
from experiments.value_lr_physical.runner import physical_value_runner
from masked_ttt import MaskedTTT, Request
import masked_ttt as runtime
from test_long_rollout_steps_extend import fixture_protocol
from test_materialization_equivalence import gpt_gqa_engine
from test_public_linked_selection import _source
from test_standalone import make_engine


def fixture_value_development():
    pilot, _ = fixture_protocol()
    development = dict(case_count=200, case_ids=pilot["development_case_ids"], holdout_count=800,
        holdout_case_ids=pilot["holdout_case_ids"], value_learning_rates=[.0001, .0003, .001],
        keep_fraction=.1, gain_learning_rate=.01, basis_source="document", selected_transfer_weight=0.,
        reference_answers_used=False, model_protocols={"gpt": "protocol.json"}, random_order_seed=1)
    names = ["whole", "zero", "teacher"] + [name_for_lr(x) for x in development["value_learning_rates"]]
    rows = [dict(case=case, **{name: 0 for name in names}) for case in development["case_ids"]]
    for row in rows[:100]:
        row.update({name: 1 for name in names})
    for row in rows[100:102]:
        row["lr_0.0003"] = row["lr_0.0010"] = 1
    rows[102]["lr_0.0010"] = 1
    accuracy = {name: sum(row[name] for row in rows) / 2 for name in names}
    choice = dict(value_learning_rate=.001, reason="predeclared_development_rule",
                  development_accuracy=51.5, control_threshold=50., holdout_evaluated=False)
    report = dict(metric="answer_focused_lexical_v2", expected=200, development_only=True,
        evaluation_answers_used_in_training=False, online_api_calls=0, same_teacher_student_kl=True,
        same_document_basis=True, selected_transfer_weight=0., rates=development["value_learning_rates"],
        errors=[], invalid=[], models={"gpt": dict(rows=rows, overall=dict(n=200, accuracy=accuracy), choice=choice)})
    profile = dict(student_rollout_tokens=8, steps=2, value_learning_rate=.0001, gain_learning_rate=.01,
        teacher_response_instruction="Return only the final answer in the required format. Do not add an explanation.")
    return development, report, profile


def test_only_completed_model_selects_rate_and_reserves_disjoint_holdout():
    development, report, profile = fixture_value_development()
    report["errors"] = [{"model": "qwen", "case": "unrelated"}]
    choice = development_choice(development, report, "gpt")
    manifest = holdout_manifest(development, report, profile, "gpt")
    assert choice["value_learning_rate"] == manifest["fixed_value_learning_rate"] == .001
    assert manifest["models"] == ["gpt"] and manifest["case_count"] == 800
    assert not set(manifest["case_ids"]) & set(manifest["development_case_ids"])
    assert not manifest["holdout_evaluated_at_selection"]


@pytest.mark.parametrize("defect", ["partial", "invalid", "stale_summary", "changed_choice"])
def test_unclean_or_incomplete_model_development_cannot_promote(defect):
    development, report, _ = fixture_value_development()
    if defect == "partial":
        report["models"]["gpt"]["rows"].pop()
    elif defect == "invalid":
        report["invalid"].append({"model": "gpt"})
    elif defect == "stale_summary":
        report["models"]["gpt"]["overall"]["accuracy"]["teacher"] += 1
    else:
        report["models"]["gpt"]["choice"]["value_learning_rate"] = .0003
    with pytest.raises(ValueError):
        development_choice(development, report, "gpt")


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_fixed_lr_uses_full_student_and_releases_standalone_cache(tmp_path, architecture):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.eos_ids = {-1}
    impl = SimpleNamespace(ROOT="old", replace=replace, validate=lambda *_args: None,
                           validate_manifest=lambda value: value)
    old = (impl.ROOT, impl.replace, impl.validate, impl.validate_manifest,
           paired_controls.original_question_controls)
    with physical_value_runner(impl, paired_controls, root="holdout", fixed_lr=.001,
                               physical_controls=kv_only_original_question_controls):
        config = impl.replace(engine.config, max_context=2048, keep_fraction=.1, steps=2,
            student_rollout_tokens=8, self_question_count=0, guard_enabled=True,
            selected_transfer_weight=0., value_basis_source="document",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        assert config.value_learning_rate == .001 and config.gain_head_mode == "kv"
        engine.config = config
        selector = runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with paired_controls.original_question_controls():
            result = MaskedTTT(engine, selector).answer(Request(
                document=document, question=question, max_new_tokens=4))
        impl.validate(result, {"key": "case"})
        assert result["training"]["fixed_value_learning_rate"] == .001
        assert result["training"]["objective"] == "on-policy KL(teacher || student)"
        assert result["training"]["selected_transfer_weight"] == 0
        assert result["training"]["teacher_conditioned_on_student_prefix"]
        assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
        assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
        memory = result["standalone_cartridge"]["memory"]
        assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0
        assert memory["additional_source_kv_bytes_before_decode"] == 0
        assert memory["additional_training_adapter_bytes_before_decode"] == 0
        model = "qwen" if architecture == "qwen3" else "gpt"
        model_id = "Qwen/Qwen3-4B-Instruct-2507" if model == "qwen" else "openai/gpt-oss-20b"
        result.update(key="case", model=model_id, task="qa1", keep_fraction=.1,
                      request_chat_date="2026-11-03")
        validate_result(result, "case", model, "qa1", .001)
    assert (impl.ROOT, impl.replace, impl.validate, impl.validate_manifest,
            paired_controls.original_question_controls) == old
