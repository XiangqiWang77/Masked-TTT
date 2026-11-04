from pathlib import Path

import pytest

from experiments.value_lr_physical.canonical_memory_hook import install


def test_hooks_attach_to_existing_hourly_services_without_new_jobs(tmp_path):
    unit, workspace = tmp_path / "units", tmp_path / "workspace"
    script = tmp_path / "canonical_memory_report.py"
    script.write_text("# frozen script\n")
    install(unit, script, workspace, reload=False)
    for stage in ("dev", "holdout"):
        file = unit / f"query-cartridges-canonical-{stage}.service.d/memory-report.conf"
        text = file.read_text()
        assert text.startswith("[Service]\nExecStartPost=/usr/bin/python3 ")
        assert str(script) in text and f"canonical_pair_{stage}_20261104" in text
        assert "--candidate-root" in text and "--output" in text
    install(unit, script, workspace, reload=False)
    assert len(list(unit.glob("**/*.conf"))) == 2


def test_missing_or_unsafe_script_paths_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="required"):
        install(tmp_path / "units", tmp_path / "missing.py", tmp_path, reload=False)
    unsafe = tmp_path / "has space"
    unsafe.write_text("# script\n")
    with pytest.raises(ValueError, match="simple paths"):
        install(tmp_path / "units", unsafe, tmp_path, reload=False)
    with pytest.raises(ValueError, match="simple paths"):
        install(Path("relative"), tmp_path / "valid.py", tmp_path, reload=False)
