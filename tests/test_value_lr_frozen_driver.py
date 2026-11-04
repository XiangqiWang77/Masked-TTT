"""Exercise the submitted driver on synthetic cases without a GPU or scheduler.

The optional driver fixture is supplied by the frozen-job preparation. No
real evaluation text, labels, GPU allocation, API call or Slurm action is used.
"""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.value_lr_physical.report import validate_result
from experiments.value_lr_physical.runner import physical_value_runner
import experiments.anchored_short_verifier.paired_controls as paired_controls
from masked_ttt import Backbone
from test_materialization_equivalence import gpt_gqa_engine
from test_public_linked_selection import _source
from test_standalone import make_engine


@pytest.mark.parametrize("model", ["qwen", "gpt"])
def test_frozen_driver_runs_a_real_request_and_releases_all_training_state(tmp_path, monkeypatch, model):
    driver_env = os.environ.get("CARTRIDGE_VALUE_DRIVER_PATH")
    if not driver_env:
        pytest.skip("A frozen physical Value-LR driver fixture was not supplied")
    driver = Path(driver_env).resolve()
    original_root = driver.parent.parent
    manifest = json.loads((original_root / "manifest.json").read_text())
    base_steps = int(os.environ.get("CARTRIDGE_TEST_BASE_STEPS", "2"))
    if base_steps not in (2, 8):
        raise ValueError("The frozen driver test supports the two submitted budgets")
    if base_steps == 8:
        manifest["steps"] = 8
        manifest["fixed_value_learning_rate"] = .0001
    fixed_lr = manifest["fixed_value_learning_rate"]
    profile = json.loads((driver.parent / "olmo_profile.json").read_text())
    profile["self_question_count"] = 0
    manifest["models"] = [model]
    order = list(manifest["case_ids"])
    random.Random(manifest["random_order_seed"]).shuffle(order)
    first = order[0]

    job = tmp_path / ("synthetic-" + model)
    runtime_root = job / "runtime"
    (runtime_root / "helpers").mkdir(parents=True)
    (job / "report.py").write_text("# Simulated report; no evaluation labels.\n")
    (job / "manifest.json").write_text(json.dumps(manifest))
    (runtime_root / "olmo_profile.json").write_text(json.dumps(profile))
    helper_source = driver.parent / "helpers/time_budget.py"
    (runtime_root / "helpers/time_budget.py").write_text(helper_source.read_text())
    scores = tmp_path / ("scores-" + model)
    scores.mkdir()
    document, question, _, _ = _source(scores)
    score_path = scores / "scores/case.json"
    score = json.loads(score_path.read_text())
    score["case_key"] = first
    score_path.write_text(json.dumps(score))
    (scores / "case_to_score_file.json").write_text(json.dumps({first: "case.json"}))
    synthetic_case = dict(key=first, document=document, query=question,
                          prompt="Read this document:\n" + document + "\nQuestion: " + question,
                          max_new_tokens=3)
    source_bytes = (json.dumps(synthetic_case) + "\n").encode()
    source_path = job / "source.jsonl"
    source_path.write_bytes(source_bytes)
    index = [dict(key=case, offset=0, bytes=len(source_bytes), task=case.split("-", 3)[2],
                  length_bucket=int(case.split("-")[1]) * 1024) for case in manifest["case_ids"]]
    index_path = job / "index.json"
    index_path.write_text(json.dumps(index))
    spec_path = job / "protocol.json"
    spec_path.write_text(json.dumps(dict(model_path=str(job / "model"),
        model_id=("Qwen/Qwen3-4B-Instruct-2507" if model == "qwen" else "openai/gpt-oss-20b"),
        source_cases=str(source_path), index=str(index_path), shared_scores=str(scores))))
    manifest["model_protocols"] = {model: str(spec_path)}
    (job / "manifest.json").write_text(json.dumps(manifest))

    # The driver changes its import paths and disables online selection.
    # Isolate those process-global changes from the rest of the test suite.
    monkeypatch.setattr(sys, "path", list(sys.path))
    from masked_ttt.decision_api import DecisionClient
    monkeypatch.setattr(DecisionClient, "__init__", DecisionClient.__init__)
    spec = importlib.util.spec_from_file_location("synthetic_frozen_driver_" + model, driver)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    implementation.ROOT, implementation.CROSS, implementation.FROZEN = job, tmp_path, runtime_root
    implementation.REPO = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(implementation.signal, "signal", lambda *_args: None)
    monkeypatch.syspath_prepend(str(runtime_root / "helpers"))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURMD_NODENAME", "synthetic-node")
    monkeypatch.setenv("MASKED_TTT_FIXED_CHAT_DATE", "2026-11-03")
    engine = make_engine() if model == "qwen" else gpt_gqa_engine()
    engine.eos_ids = {-1}

    def fake_load(_cls, _path, config=None, **_kwargs):
        engine.config = config
        engine.cfg = dict(max_context=config.max_context, chunk_tokens=config.chunk_tokens,
                          chunk_boundary_mode="legacy", prefill_chunk_tokens=config.prefill_chunk_tokens)
        return engine

    monkeypatch.setattr(Backbone, "load", classmethod(fake_load))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "synthetic-gpu")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _index: SimpleNamespace(total_memory=96 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def fake_subprocess(command, **_kwargs):
        calls.append(command)
        if command[:3] == ["scontrol", "show", "job"]:
            return SimpleNamespace(returncode=0, stdout="TimeLimit=06:00:00 RunTime=00:00:00", stderr="")
        if command[:2] == ["scontrol", "requeue"] or command[:2] == ["scontrol", "update"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if len(command) == 2 and command[1] == str(job / "report.py"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError("The probe attempted an unexpected external command: " + str(command))

    monkeypatch.setattr(implementation.subprocess, "run", fake_subprocess)
    original_prepare = implementation.prepare_case
    requests = []

    def first_request_only(*args):
        if requests:
            raise TimeoutError("Stop the synthetic driver after one case")
        result = original_prepare(*args)
        requests.append(result)
        return result

    monkeypatch.setattr(implementation, "prepare_case", first_request_only)
    if base_steps == 8:
        from experiments.step_physical.runner import physical_step_runner
        context = physical_step_runner(implementation, paired_controls, root=job, fixed_steps=8,
                                       physical_controls=kv_only_original_question_controls)
    else:
        context = physical_value_runner(implementation, paired_controls, root=job, fixed_lr=fixed_lr,
                                        physical_controls=kv_only_original_question_controls)
    with context:
        implementation.main()

    result = json.loads((job / "cases" / model / first / "result.json").read_text())
    validate_result(result, first, model, first.split("-", 3)[2], fixed_lr, base_steps=base_steps)
    assert result["prediction"] == result["standalone_cartridge"]["prediction"]
    assert engine.config.value_learning_rate == fixed_lr and engine.config.gain_learning_rate == .01
    assert engine.config.gain_head_mode == "kv" and engine.config.steps == base_steps
    if base_steps == 8:
        probe = result["standalone_cartridge"]["step_probe"]
        assert result["training"]["fixed_base_steps"] == 8
        assert probe["checkpoints"]["8"]["tokens"] == result["generated_token_ids"]
        assert probe["diagnostic_snapshots_released_before_final_decode"]
    assert result["training"]["student_rollout_limit"] == 8
    assert result["training"]["fixed_value_learning_rate"] == fixed_lr
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    assert result["standalone_cartridge"]["memory"]["additional_source_kv_bytes_before_decode"] == 0
    assert result["standalone_cartridge"]["memory"]["additional_training_adapter_bytes_before_decode"] == 0
    assert any(command[:2] == ["scontrol", "requeue"] for command in calls)
