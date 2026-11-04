from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import sys

import pytest
import torch

from test_materialization_equivalence import gpt_gqa_engine
from test_standalone import make_engine


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "experiments/kv_only_cartridge/in_job_probe.py"
CANDIDATE = ROOT / "src/masked_ttt/pipeline.py"
CONTROL = ROOT / "experiments/kv_only_cartridge/paired_control.py"


def load_probe(path):
    spec = spec_from_file_location("candidate_in_job_probe", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_candidate_matches_frozen_model_and_restores_state(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    config, state = engine.config, torch.get_rng_state()
    observed = load_probe(PROBE).validate_loaded_candidate(engine, CANDIDATE)
    assert observed["passed"] and observed["same_training_gradients"]
    assert observed["same_standalone_logits"] and observed["same_standalone_tokens"]
    assert observed["no_full_kv_or_extra_adapter_at_decode"]
    assert engine.config is config and torch.equal(torch.get_rng_state(), state)
    assert "masked_ttt._request_local_validation_pipeline" not in sys.modules


def test_one_model_probe_is_recorded_once(tmp_path):
    import json

    engine = make_engine()
    shutil.copy2(PROBE, tmp_path / "driver.py")
    shutil.copy2(CANDIDATE, tmp_path / "pipeline_candidate.py")
    shutil.copy2(CONTROL, tmp_path / "paired_control_candidate.py")
    (tmp_path / "ready.json").write_text(json.dumps({"allow_single_loaded_model_probe": True}))
    probe = load_probe(tmp_path / "driver.py")
    config, state = engine.config, torch.get_rng_state()
    assert probe.run_if_needed(engine, "qwen", {"gpu": "cpu-test"}) == "passed"
    assert probe.run_if_needed(engine, "qwen", {"gpu": "cpu-test"}) == "already_passed"
    report = json.loads((tmp_path / "qwen.json").read_text())
    assert report["passed"] and report["same_standalone_logits"]
    assert report["same_generic_tokens_before_after_release"]
    assert report["original_benchmark_runtime_unchanged"]
    assert engine.config is config and torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_generic_format_survives_source_release(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    config, state = engine.config, torch.get_rng_state()
    result = load_probe(PROBE).validate_loaded_generic_format(engine, CONTROL)
    assert result["same_generic_tokens_before_after_release"]
    assert not result["generic_teacher_answer_in_prompt"]
    assert engine.config is config and torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss_gqa"])
def test_position_check_uses_actual_local_checkpoint_config(architecture, tmp_path):
    engine = make_engine() if architecture == "qwen3" else gpt_gqa_engine()
    engine.model.config.save_pretrained(tmp_path)
    engine.model.name_or_path = str(tmp_path)
    probe = load_probe(PROBE)
    result = probe.validate_loaded_checkpoint(engine)
    assert result["same_checkpoint_rope"] and result["native_checkpoint_config_checked"]
    bad = engine.model.config.to_dict()
    bad["rope_theta"] *= 2
    import json
    (tmp_path / "config.json").write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        probe.validate_loaded_checkpoint(engine)


def test_failed_candidate_restores_configuration_and_rng(tmp_path):
    import masked_ttt.pipeline as pipeline

    engine = make_engine()
    config, state = engine.config, torch.get_rng_state()
    hooks = (pipeline.answer_standalone, pipeline.decode_standalone)
    with pytest.raises(FileNotFoundError):
        load_probe(PROBE).validate_loaded_candidate(engine, tmp_path / "missing.py")
    assert engine.config is config and torch.equal(torch.get_rng_state(), state)
    assert (pipeline.answer_standalone, pipeline.decode_standalone) == hooks
    assert "masked_ttt._request_local_validation_pipeline" not in sys.modules


def test_gpt_pair_pins_one_chat_date_across_both_calls(monkeypatch):
    import masked_ttt.architectures as architectures

    dates = []
    def clock():
        dates.append("read")
        return "2026-11-03" if len(dates) == 1 else "2026-11-04"
    monkeypatch.setattr(architectures, "_template_date", clock)
    engine = gpt_gqa_engine()
    assert load_probe(PROBE).validate_loaded_candidate(engine, CANDIDATE)["passed"]
    assert dates == ["read"] and engine.tokenizer.masked_ttt_request_date is None
