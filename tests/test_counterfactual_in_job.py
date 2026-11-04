import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from experiments.counterfactual_transfer import enable_in_job_probe as enable
from experiments.counterfactual_transfer import in_job_probe as probe


def valid_result():
    return dict(selected_inputs_identical=True,candidates_used_after_training_only=True,
        teacher_answer_in_final_prompt=False,whole_cache_released_before_cartridge=True,
        source_cache_released_before_scoring=True,training_adapter_released_before_scoring=True,
        probabilities_from_materialized_kv=True,margin_difference_garden_minus_office_context=.2)


def mock_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,"get_rng_state_all",lambda:[torch.tensor([1,2])])
    monkeypatch.setattr(torch.cuda,"set_rng_state_all",lambda _state:None)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)


def test_in_job_probe_restores_config_and_rng_and_is_idempotent(tmp_path,monkeypatch):
    mock_cuda(monkeypatch)
    config=object();engine=SimpleNamespace(device=torch.device("cuda:0"),config=config)
    original=torch.get_rng_state().clone();calls=[]
    def run(actual,**kwargs):
        calls.append(kwargs);actual.config="temporary";torch.manual_seed(87)
        return valid_result()
    monkeypatch.setattr(probe,"_load_probe",lambda _root:SimpleNamespace(run_pair=run))
    assert probe.run_if_needed(engine,"qwen",dict(gpu="test"),_root=tmp_path)=="validated"
    saved=json.loads((tmp_path/"qwen_result.json").read_text())
    assert saved["probe_validated"] and saved["extra_gpu_jobs"]==0 and saved["online_api_calls"]==0
    assert saved["probabilities_from_materialized_kv"] and not saved["training_runtime_changed"]
    assert engine.config is config and torch.equal(torch.get_rng_state(),original)
    assert calls==[dict(steps=2,max_new_tokens=8,rollout_tokens=8,time_budget_seconds=90)]
    assert probe.run_if_needed(engine,"qwen",{},_root=tmp_path)=="previously_validated"
    assert len(calls)==1


def test_failures_restore_state_and_stop_after_two_attempts(tmp_path,monkeypatch):
    mock_cuda(monkeypatch)
    config=object();engine=SimpleNamespace(device=torch.device("cuda:0"),config=config)
    original=torch.get_rng_state().clone()
    def fail(actual,**_kwargs):
        actual.config="temporary";torch.manual_seed(93)
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(probe,"_load_probe",lambda _root:SimpleNamespace(run_pair=fail))
    for attempt in (1,2):
        with pytest.raises(RuntimeError,match="synthetic failure"):
            probe.run_if_needed(engine,"gpt",{},_root=tmp_path)
        assert json.loads((tmp_path/"gpt_failure.json").read_text())["attempts"]==attempt
        assert engine.config is config and torch.equal(torch.get_rng_state(),original)
    assert probe.run_if_needed(engine,"gpt",{},_root=tmp_path)=="two_previous_failures"
    with pytest.raises(ValueError,match="allocated GPU"):
        probe.run_if_needed(SimpleNamespace(device=torch.device("cpu")),"gpt",{},_root=tmp_path)


def test_enable_uses_compatible_frozen_runtime_and_never_replaces_other_hook(tmp_path):
    root=tmp_path/enable.TARGET
    frozen=root/"runtime/repo/src/masked_ttt"
    shutil.copytree(enable.REPO/"src/masked_ttt",frozen,
                    ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    (root/"runtime/rollout_driver.py").write_text("disabled_optional_probe")
    (root/"submission.json").write_text('{"job_id":"synthetic"}')
    result=enable.enable(tmp_path,root)
    assert result["enabled"] and result["extra_gpu_jobs"]==0
    assert not result["training_runtime_changed"] and result["runs_only_if_allocation_reaches_optional_hook"]
    central=tmp_path/enable.CENTRAL
    assert json.loads((central/"protocol.json").read_text())["scores_after_full_and_source_release"]
    assert (root/"disabled_optional_probe/driver.py").is_file()
    assert enable.enable(tmp_path,root)==result
    (frozen/"config.py").write_text("changed")
    with pytest.raises(ValueError,match="supporting runtime"):
        enable.enable(tmp_path,root)
    assert (central/"probe.py").is_file()
