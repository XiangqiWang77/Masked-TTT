from pathlib import Path
import shutil

import pytest

from experiments.kv_only_cartridge import enable_reusable_probe as setup


def roots(workspace, monkeypatch):
    monkeypatch.setattr(setup, "TARGETS", ("qwen-job", "gpt-job"))
    for name in setup.TARGETS:
        root = workspace / name
        (root / "runtime").mkdir(parents=True)
        (root / "runtime/rollout_driver.py").write_text("disabled_optional_probe\n")
        frozen = root / "runtime/repo"
        (frozen / "src/masked_ttt").mkdir(parents=True)
        (frozen / "experiments/kv_only_cartridge").mkdir(parents=True)
        for file in setup.SUPPORT:
            shutil.copy2(setup.REPO / "src/masked_ttt" / file, frozen / "src/masked_ttt" / file)
        shutil.copy2(setup.REPO / "experiments/kv_only_cartridge/in_job_probe.py",
                     frozen / "experiments/kv_only_cartridge/in_job_probe.py")


def test_install_is_atomic_idempotent_and_adds_no_gpu_job(tmp_path, monkeypatch):
    roots(tmp_path, monkeypatch)
    state = setup.enable(tmp_path)
    assert state["enabled"] and state["new_gpu_jobs"] == 0 and not state["training_runtime_changed"]
    assert state["runs_only_when_existing_allocation_reaches_hook"]
    for name in setup.TARGETS:
        assert (tmp_path / name / "disabled_optional_probe/ready.json").is_file()
    assert (tmp_path / setup.CENTRAL / "cartridge_candidate.py").is_file()
    assert setup.enable(tmp_path) == state


def test_rejects_existing_unrelated_hook_before_writing_anything(tmp_path, monkeypatch):
    roots(tmp_path, monkeypatch)
    existing = tmp_path / "gpt-job/disabled_optional_probe"
    existing.mkdir()
    (existing / "driver.py").write_text("unrelated work\n")
    with pytest.raises(ValueError, match="different configuration"):
        setup.enable(tmp_path)
    assert (existing / "driver.py").read_text() == "unrelated work\n"
    assert not (tmp_path / "qwen-job/disabled_optional_probe").exists()
    assert not (tmp_path / setup.CENTRAL).exists()


def test_rejects_supporting_runtime_drift(tmp_path, monkeypatch):
    roots(tmp_path, monkeypatch)
    (tmp_path / "gpt-job/runtime/repo/src/masked_ttt/cache.py").write_text("different\n")
    with pytest.raises(ValueError, match="supporting module differs"):
        setup.enable(tmp_path)
    assert not (tmp_path / setup.CENTRAL).exists()
