import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from experiments.value_lr_physical import canonical_job
from test_study_job import source_protocols


def sources():
    rollout, linked, profile = source_protocols()
    rollout["model_protocols"] = linked["model_protocols"] = {"qwen": "qwen", "gpt": "gpt"}
    feasibility = dict(model_calls=0, online_api_calls=0, reference_answers_read=False,
                       models={"qwen": {"buildable": 200}, "gpt": {"buildable": 200}})
    return rollout, linked, profile, feasibility


def test_freezes_same_olmo_method_and_one_gpu():
    manifest = canonical_job.make_manifest(*sources())
    assert manifest["models"] == ["qwen", "gpt"] and manifest["case_count"] == 200
    assert manifest["evidence_tokenization_comparison"] == "original_vs_canonical_fragment"
    assert manifest["online_api_calls"] == 0 and manifest["reference_answers_used_in_inference"] is False
    assert manifest["gain_head_mode"] == "kv" and manifest["selected_transfer_weight"] == 0
    for name in (canonical_job.DEV, canonical_job.HOLDOUT):
        script = canonical_job.batch_script(Path("/tmp") / name)
        assert "#SBATCH --gpus-per-task=1" in script and "#SBATCH --mem=64G" in script
        assert "value_lr_physical/canonical_execute.py" in script and "unset OPENAI_API_KEY" in script


def test_rejects_unverified_evidence_or_missing_model():
    rollout, linked, profile, feasibility = sources()
    feasibility["models"]["gpt"]["buildable"] = 199
    with pytest.raises(ValueError, match="feasibility"):
        canonical_job.make_manifest(rollout, linked, profile, feasibility)
    rollout, linked, profile, feasibility = sources()
    rollout["model_protocols"] = linked["model_protocols"] = {"qwen": "qwen"}
    with pytest.raises(ValueError, match="settings"):
        canonical_job.make_manifest(rollout, linked, profile, feasibility)


def test_submit_uses_one_gpu_slot_and_does_not_duplicate(tmp_path, monkeypatch):
    workspace, root = tmp_path, tmp_path / canonical_job.DEV
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(dict(models=["qwen", "gpt"], case_count=200)))
    (root / "cpu_validation_passed.json").write_text(json.dumps(dict(cpu_tests_passed=True)))
    (root / "job.sbatch").write_text("# test\n")
    @contextmanager
    def available():
        yield True
    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=available))
    calls = []
    monkeypatch.setattr(canonical_job.subprocess, "run", lambda args, **_kwargs:
                        calls.append(args) or SimpleNamespace(stdout="123\n"))
    result = canonical_job.submit(workspace, root)
    assert result["job_id"] == "123" and result["gpus"] == 1
    assert result["models"] == ["qwen", "gpt"] and result["cases_per_model"] == 200
    assert len(calls) == 1 and calls[0][0] == "sbatch"
    assert canonical_job.submit(workspace, root)["job_id"] == "123" and len(calls) == 1
