import pytest

from experiments.linked_value.memory_hook import install


def test_attaches_to_existing_hourly_holdout_service_idempotently(tmp_path):
    unit, workspace, script = tmp_path / "units", tmp_path / "workspace", tmp_path / "memory_report.py"
    script.write_text("# frozen script\n")
    install(unit, script, workspace, reload=False)
    path = unit / "query-cartridges-linked-value-holdout.service.d/memory-report.conf"
    text = path.read_text()
    assert text.startswith("[Service]\nExecStartPost=/usr/bin/python3 ")
    assert str(script) in text and "linked_value_gpt_holdout_20261104" in text
    assert "--job-root" in text and "--output" in text
    install(unit, script, workspace, reload=False)
    assert len(list(unit.glob("**/*.conf"))) == 1
    install(unit, script, workspace, kind="detailed", reload=False)
    detailed = unit / "query-cartridges-linked-value-holdout.service.d/detailed-report.conf"
    assert "--output " + str(workspace / "query_cartridges_linked_value_gpt_holdout_20261104/detailed_latest") in detailed.read_text()
    assert path.read_text() == text
    assert len(list(unit.glob("**/*.conf"))) == 2


def test_rejects_missing_or_unsafe_paths(tmp_path):
    with pytest.raises(ValueError, match="required"):
        install(tmp_path / "unit", tmp_path / "missing.py", tmp_path, reload=False)
    script = tmp_path / "has space"
    script.write_text("# script\n")
    with pytest.raises(ValueError, match="simple paths"):
        install(tmp_path / "unit", script, tmp_path, reload=False)
    script = tmp_path / "safe.py"
    script.write_text("# script\n")
    with pytest.raises(ValueError, match="Unknown report kind"):
        install(tmp_path / "unit", script, tmp_path, kind="unexpected", reload=False)
