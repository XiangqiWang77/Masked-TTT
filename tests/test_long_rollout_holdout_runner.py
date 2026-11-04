"""Fixed development-selected steps remain SD-only and physically standalone."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.anchored_short_verifier.paired_controls as paired_controls
import experiments.long_rollout_steps.holdout_runner as implementation_module
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.long_rollout_steps.holdout_runner import fixed_step_runner
from masked_ttt import MaskedTTT, Request
from masked_ttt.linked_selection import LinkedFrozenSelector
import masked_ttt as runtime
from test_materialization_equivalence import gpt_gqa_engine
from test_public_linked_selection import _source
from test_standalone import make_engine


@pytest.mark.parametrize("architecture,step", [("qwen3", 2), ("gpt_oss", 4)])
def test_fixed_step_is_used_by_model_and_restores_hooks(tmp_path, monkeypatch, architecture, step):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.eos_ids = {-1}
    engine.config = replace(engine.config, max_context=2048, keep_fraction=.1,
        student_rollout_tokens=32, self_question_count=0, guard_enabled=False,
        selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    model_path = tmp_path / ("fake-" + architecture)
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda value: value,
                                    validate=lambda _result, _entry: None,
                                    preflight=lambda _engine: None)
    def load(_cls, _path, config=None, **_kwargs):
        engine.config = config
        return engine
    monkeypatch.setattr(runtime.Backbone, "load", classmethod(load))
    monkeypatch.setattr(implementation_module, "checkpoint_preflight",
                        lambda _engine, final_base_step: {"passed": True, "step": final_base_step})
    original_descriptor = runtime.Backbone.__dict__["load"]
    original = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
                implementation.preflight, runtime.FrozenJSONSelector, paired_controls.original_question_controls)
    with fixed_step_runner(implementation, runtime, paired_controls, root="holdout",
            linked_selector=LinkedFrozenSelector, physical_controls=kv_only_original_question_controls,
            model_path_steps={model_path: step}):
        with pytest.raises(ValueError, match="model or configuration"):
            runtime.Backbone.load(tmp_path / "unknown", config=engine.config)
        loaded = runtime.Backbone.load(model_path, config=engine.config)
        assert loaded.config.steps == loaded.config.max_answer_steps == step
        assert implementation.preflight(loaded) == {"passed": True, "step": step}
        selector = runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with paired_controls.original_question_controls():
            result = MaskedTTT(loaded, selector).answer(Request(
                document=document, question=question, max_new_tokens=4))
        implementation.validate(result, {"key": "case"})
        assert len(result["training"]["steps"]) == step
        assert result["training"]["objective"] == "on-policy KL(teacher || student)"
        assert result["training"]["selected_transfer_weight"] == 0
        memory = result["standalone_cartridge"]["memory"]
        assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0
        assert memory["additional_source_kv_bytes_before_decode"] == 0
        assert memory["additional_training_adapter_bytes_before_decode"] == 0
        assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    assert runtime.Backbone.__dict__["load"] is original_descriptor
    assert (implementation.ROOT, implementation.validate_manifest, implementation.validate,
            implementation.preflight, runtime.FrozenJSONSelector, paired_controls.original_question_controls) == original


def test_invalid_fixed_steps_are_rejected():
    with pytest.raises(ValueError, match="steps"):
        with fixed_step_runner(None, None, None, root="x", linked_selector=None,
                physical_controls=None, model_path_steps={Path("/tmp/example"): 3}):
            pass
