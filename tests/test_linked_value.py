"""The linked-value candidate keeps the OLMo SD and independent cartridge path."""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import masked_ttt as runtime
import experiments.anchored_short_verifier.paired_controls as controls
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.linked_value.monitor import development_decision
from experiments.linked_value import monitor, report
from experiments.linked_value.prepare import DEV, HOLDOUT, batch_script
from experiments.linked_value.runner import linked_value_runner
from masked_ttt import MaskedTTT, Request, TTTConfig
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_public_linked_selection import _source
from test_standalone import make_engine


def test_fixed_protocol_and_hooks_restore_on_failure():
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x,
                                     validate=lambda *_: None, replace=replace)
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.replace, runtime.FrozenJSONSelector, controls.original_question_controls)
    manifest = dict(mode="development", models=["gpt"], selection_policy="decision_entity_link_v1",
                    entity_reserve_fraction=.25, fixed_value_learning_rate=.001, student_rollout_tokens=8,
                    gain_head_mode="kv", steps=2, max_answer_steps=8, holdout_evaluated_at_selection=False)
    with pytest.raises(RuntimeError, match="intentional"):
        with linked_value_runner(implementation, runtime, controls, root="candidate",
                                 selector=LinkedFrozenSelector, physical_controls=kv_only_original_question_controls):
            assert implementation.validate_manifest(manifest)["selection_policy"] == "cached_decision_score10"
            cfg = implementation.replace(TTTConfig(), max_context=8192)
            assert cfg.value_learning_rate == .001 and cfg.gain_head_mode == "kv" and cfg.steps == 2
            assert cfg.max_answer_steps == 8 and cfg.answer_KL_stop == .5
            assert cfg.standalone_answer_format == "original" and not cfg.align_student_teacher_prompt
            for key, value in (("fixed_value_learning_rate", .003), ("entity_reserve_fraction", .5),
                               ("max_answer_steps", 2), ("holdout_evaluated_at_selection", True)):
                with pytest.raises(ValueError):
                    implementation.validate_manifest(dict(manifest, **{key: value}))
            raise RuntimeError("intentional")
    assert (implementation.ROOT, implementation.validate_manifest, implementation.validate,
            implementation.replace, runtime.FrozenJSONSelector, controls.original_question_controls) == old


def test_linked_scores_original_sd_and_physically_released_cartridge(tmp_path, monkeypatch):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine()
    engine.eos_ids = {-1}
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x,
                                     validate=lambda *_: None, replace=replace)
    with linked_value_runner(implementation, runtime, controls, root="candidate",
                             selector=LinkedFrozenSelector, physical_controls=kv_only_original_question_controls):
        engine.config = implementation.replace(engine.config, max_context=engine.config.max_context,
            keep_fraction=.1, student_rollout_tokens=8, self_question_count=0, guard_enabled=False,
            selected_transfer_weight=0., value_basis_source="document", release_full_before_cartridge=True,
            release_selected_source_before_decode=True, teacher_response_instruction="Return only the final answer.")
        selector = runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with controls.original_question_controls():
            result = MaskedTTT(engine, selector).answer(Request(document=document, question=question, max_new_tokens=8))
        implementation.validate(result, {"key": "case"})
    assert result["training"]["objective"] == "on-policy KL(teacher || student)"
    assert result["training"]["value_learning_rate"] == .001
    assert result["training"]["gain_learning_rate"] == .01
    assert len(result["training"]["steps"]) == 2
    assert result["selection"]["selected_chunks"] == [1, 2, 3, 6]
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    memory = result["standalone_cartridge"]["memory"]
    assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0
    assert memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0

    # Exercise the actual report on a generated result, including strict pairing.
    result.update(key="case", model="test", model_short="gpt", task="qa1", length=8192,
                  keep_fraction=.1, request_chat_date="2026-11-03")
    result["self_judge"]["native_prediction"] = "blue"
    old = deepcopy(result)
    controls_before = old["standalone_cartridge"]["original_question_controls"]
    old["standalone_cartridge"]["paired_prompt_control"] = dict(
        zero_original=controls_before["zero"], trained_original=controls_before["adapted"],
        same_selected_kv=True, same_generation_limit=True, reference_answers_used=False)
    for folder, value in ((tmp_path / "candidate", result), (tmp_path / "baseline", old)):
        path = folder / "cases/gpt/case/result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    development = ["case"] + [f"missing-{i}" for i in range(199)]
    holdout = [f"held-{i}" for i in range(800)]
    manifest = dict(mode="development", models=["gpt"], case_count=200, case_ids=development,
        development_case_ids=development, holdout_case_ids=holdout, selection_policy="decision_entity_link_v1",
        entity_reserve_fraction=.25, keep_fraction=.1, fixed_value_learning_rate=.001,
        student_rollout_tokens=8, gain_head_mode="kv", max_answer_steps=8, selected_transfer_weight=0.,
        reference_answers_used_in_inference=False, online_api_calls=0, holdout_evaluated_at_selection=False)
    monkeypatch.setattr(report, "validate_partition", lambda *_: None)
    stats = report.collect(manifest, tmp_path / "candidate", tmp_path / "baseline",
        {"case": dict(task="qa1", answers=["blue"])}, lambda text, *_: (int("blue" in text),))
    assert not stats["invalid"] and not stats["errors"] and not stats["unmatched_baselines"]
    assert stats["overall"]["n"] == 1 and stats["overall"]["strictly_paired"]["n"] == 1
    assert not development_decision(stats)["eligible"]


def test_promotion_requires_all_200_and_each_paired_control():
    report = dict(mode="development", expected=200, metric="answer_focused_lexical_v2", errors=[], invalid=[],
        overall=dict(n=200, accuracy=dict(whole=50., zero=52., teacher=54., cartridge=56.),
                     strictly_paired=dict(n=180, delta_pp=1., new_correct=3, new_wrong=1)))
    assert development_decision(report)["eligible"]
    report["overall"]["accuracy"]["teacher"] = 56.
    assert not development_decision(report)["eligible"]
    report["overall"]["accuracy"]["teacher"] = 54.
    report["invalid"].append({})
    assert not development_decision(report)["eligible"]


def test_development_and_holdout_scripts_request_one_suitable_gpu():
    for name in (DEV, HOLDOUT):
        script = batch_script(Path("/tmp") / name)
        assert "#SBATCH --gpus-per-task=1" in script
        assert "#SBATCH --mem=64G" in script and "#SBATCH --time-min=00:20:00" in script
        assert "linked_value/execute.py" in script and "gpu:a40" not in script


def test_combined_report_preserves_source_split_and_previously_viewed_cases():
    dev_sources = [f"qa1-{900000+i}" for i in range(50)]
    held_sources = ["qa1-300023", "qa2-300037"] + [f"qa2-{800000+i}" for i in range(198)]
    def rows(sources):
        return [dict(case=f"babilong-{length}-{source}", task="qa1", length=int(length)*1024,
                     whole=0, zero=0, teacher=0, cartridge=1, baseline=None, cartridge_percent=12.)
                for source in sources for length in ("8", "16", "32", "64")]
    dev, held = rows(dev_sources), rows(held_sources)
    manifest = dict(development_case_ids=[r["case"] for r in dev],
                    holdout_case_ids=[r["case"] for r in held],
                    qualitative_case_sources=["qa1-300023", "qa2-300037"])
    result = monitor.combine(manifest, dict(metric="answer_focused_lexical_v2", errors=[], invalid=[], rows=dev),
                             dict(metric="answer_focused_lexical_v2", errors=[], invalid=[], rows=held))
    assert result["overall"]["n"] == 1000 and result["holdout"]["n"] == 800
    assert result["holdout_excluding_two_previously_viewed_sources"]["n"] == 792
