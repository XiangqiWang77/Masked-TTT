from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from experiments.value_lr_physical.prepare_gain_transfer import make_manifest, make_slurm, submit


def create_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    dev = workspace / "query_cartridges_value_lr_20261104"
    anchor = workspace / "query_cartridges_basis_dev_20261104"
    dev.mkdir(parents=True)
    anchor.mkdir()
    keys = [f"babilong-{length}-qa{i}-300000" for i in range(10) for length in (8, 16, 32, 64)]
    (anchor / "manifest.json").write_text(json.dumps(dict(case_ids=keys)))
    (dev / "manifest.json").write_text(json.dumps(dict(case_ids=keys, keep_fraction=.1,
        reference_answers_used=False, selected_transfer_weight=0, model_protocols=dict(qwen="q", gpt="g"))))
    (dev / "olmo_profile.json").write_text(json.dumps(dict(steps=2, student_rollout_tokens=8,
        value_learning_rate=.0001, gain_learning_rate=.01)))
    root = workspace / "query_cartridges_gain_transfer_20261104"
    return workspace, root


def test_manifest_freezes_original_method_and_source_aligned_cases(tmp_path):
    workspace, _ = create_workspace(tmp_path)
    manifest, _ = make_manifest(workspace)
    assert manifest["models"] == ["qwen", "gpt"] and len(manifest["case_ids"]) == 40
    assert manifest["keep_fraction"] == .1 and manifest["selected_transfer_weight"] == 0
    assert manifest["online_api_calls"] == 0 and manifest["diagnosis_only"]
    assert manifest["value_alignment_diagnostic"] is False
    aligned, _ = make_manifest(workspace, value_alignment=True)
    assert aligned["value_alignment_diagnostic"] is True
    combined, _ = make_manifest(workspace, value_alignment=True, zero_path_parity=True)
    assert combined["zero_path_parity_diagnostic"] is True
    with pytest.raises(ValueError):
        make_manifest(workspace, zero_path_parity=True)
    path = workspace / "query_cartridges_value_lr_20261104/olmo_profile.json"
    profile = json.loads(path.read_text())
    profile["student_rollout_tokens"] = 32
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        make_manifest(workspace)


def test_script_requests_one_low_priority_compatible_gpu(tmp_path):
    _, root = create_workspace(tmp_path)
    script = make_slurm(root)
    assert "#SBATCH --gpus-per-task=1" in script
    assert "#SBATCH --nice=10000" in script
    assert "#SBATCH --partition=gpu_h200,gpu_h100,gpu_b200,gpu_rtx6000,scavenge_gpu" in script
    assert "gpu:h100|gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell" in script
    assert "unset OPENAI_API_KEY OPENAI_API_KEY_FILE" in script
    assert "value_lr_physical/run_gain_transfer.py" in script
    assert "--rootdir" in script and "--confcutdir" in script
    assert "--value-alignment" not in script
    assert "--value-alignment" in make_slurm(root, value_alignment=True)
    combined = make_slurm(root, value_alignment=True, zero_path_parity=True)
    assert "--value-alignment --zero-path-parity" in combined


def test_submission_obeys_gpu_cap_and_is_idempotent(tmp_path, monkeypatch):
    import sys
    import experiments.value_lr_physical.prepare_gain_transfer as module

    workspace, root = create_workspace(tmp_path)
    root.mkdir()
    (root / "cpu_validation_passed.json").write_text(json.dumps(dict(cpu_tests_passed=True)))
    (root / "manifest.json").write_text(json.dumps(dict(value_alignment_diagnostic=False)))
    (root / "job.sbatch").write_text("test")
    available = [True]

    @contextmanager
    def slot():
        yield available[0]

    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=slot))
    calls = []

    def launch(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="123456\n")

    monkeypatch.setattr(module.subprocess, "run", launch)
    available[0] = False
    with pytest.raises(RuntimeError, match="12-GPU"):
        submit(workspace, root)
    assert not calls and not (root / "submission.json").exists()
    available[0] = True
    first = submit(workspace, root)
    second = submit(workspace, root)
    assert first == second and first["job_id"] == "123456" and len(calls) == 1
