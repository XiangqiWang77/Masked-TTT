import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from experiments.counterfactual_transfer import enable_in_job_probe as enable
from experiments.counterfactual_transfer import enable_hourly_report as hourly
from experiments.counterfactual_transfer import in_job_probe as probe
from experiments.counterfactual_transfer import report_in_job as report
from test_standalone import make_engine


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


def test_probe_bypasses_benchmark_package_method_wrapper(tmp_path,monkeypatch):
    """The paired benchmark temporarily replaces masked_ttt.MaskedTTT."""
    import masked_ttt
    engine=make_engine()
    class BenchmarkWrapper:
        def __init__(self,*_args,**_kwargs):
            raise AssertionError("The benchmark wrapper must not run the diagnostic")
    monkeypatch.setattr(masked_ttt,"MaskedTTT",BenchmarkWrapper)
    shutil.copy2(enable.REPO/"experiments/counterfactual_transfer/probe.py",tmp_path/"probe.py")
    loaded=probe._load_probe(tmp_path)
    result=loaded.run_pair(engine,steps=1,max_new_tokens=2,rollout_tokens=2)
    assert result["selected_inputs_identical"] and result["probabilities_from_materialized_kv"]


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
    path=central/"probe.py";new=path.read_bytes()
    new_import=b"from masked_ttt.pipeline import MaskedTTT\nfrom masked_ttt.request import Request\n"
    old_import=b"from masked_ttt import MaskedTTT,Request\n"
    assert new.count(new_import)==1
    path.write_bytes(new.replace(new_import,old_import,1))
    upgraded=enable.upgrade_method_binding(tmp_path,root)
    assert upgraded["updated"] and path.read_bytes()==new
    assert enable.upgrade_method_binding(tmp_path,root)["already_current"]
    path.write_bytes(b"unexpected code")
    with pytest.raises(ValueError,match="expected previous version"):
        enable.upgrade_method_binding(tmp_path,root)
    path.write_bytes(new)
    (frozen/"config.py").write_text("changed")
    with pytest.raises(ValueError,match="supporting runtime"):
        enable.enable(tmp_path,root)
    assert (central/"probe.py").is_file()


def test_hourly_report_handles_pending_failure_and_valid_models(tmp_path,monkeypatch):
    first=report.collect(tmp_path)
    assert not first["all_models_valid"] and first["models"]["qwen"]["status"]=="pending"
    (tmp_path/"gpt_failure.json").write_text(json.dumps(dict(attempts=2,error_type="RuntimeError")))
    assert report.collect(tmp_path)["models"]["gpt"]["status"]=="failed"
    for model in ("qwen","gpt"):
        data=dict(valid_result(),probe_validated=True,model=model,gpu="test",timestamp=1.,
            reference_answers_used=False,online_api_calls=0,extra_gpu_jobs=0,training_runtime_changed=False,
            zero_cache_max_logprob_difference=0.,dynamic_physical_max_abs_logprob_difference=0.,
            state_difference_norms=dict(gain=.1,value_coefficients=.2,fixed_value_basis=.3),
            full_baseline_mentions_expected_for_both=True,cartridge_mentions_expected_for_both=False)
        (tmp_path/(model+"_result.json")).write_text(json.dumps(data))
    data=report.collect(tmp_path)
    assert data["all_models_valid"] and data["models"]["gpt"]["status"]=="valid"
    assert "0.200000" in report.render(data) and "general accuracy" in report.render(data)
    monkeypatch.setattr("sys.argv",["report","--results-root",str(tmp_path)])
    report.main()
    assert json.loads((tmp_path/"latest.json").read_text())["all_models_valid"]
    assert (tmp_path/"latest.md").is_file()
    bad=json.loads((tmp_path/"qwen_result.json").read_text());bad["model"]="gpt"
    (tmp_path/"qwen_result.json").write_text(json.dumps(bad))
    assert report.collect(tmp_path)["models"]["qwen"]["status"]=="invalid"


def test_hourly_report_installs_without_overwriting_existing_service(tmp_path):
    root=tmp_path/enable.TARGET
    frozen=root/"runtime/repo/src/masked_ttt"
    shutil.copytree(enable.REPO/"src/masked_ttt",frozen,
                    ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    (root/"runtime/rollout_driver.py").write_text("disabled_optional_probe")
    (root/"submission.json").write_text('{"job_id":"synthetic"}')
    enable.enable(tmp_path,root)
    units=tmp_path/"units";units.mkdir()
    (units/hourly.SERVICE).write_text("[Service]\nExecStart=/usr/bin/python3 "+str(root)+"\n")
    installed=hourly.install(tmp_path,root,unit_root=units,reload=False)
    assert installed["installed"] and installed["extra_gpu_jobs"]==0
    assert installed["reporting_interval"]=="existing hourly timer"
    dropin=Path(installed["dropin"])
    assert "ExecStartPost=/usr/bin/python3" in dropin.read_text()
    assert hourly.install(tmp_path,root,unit_root=units,reload=False)==installed
    dropin.write_text("different")
    with pytest.raises(ValueError,match="different content"):
        hourly.install(tmp_path,root,unit_root=units,reload=False)
    assert dropin.read_text()=="different"
