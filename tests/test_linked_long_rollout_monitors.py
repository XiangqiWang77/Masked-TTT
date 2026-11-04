from pathlib import Path

import pytest

from experiments.babilong_linked_selection.long_rollout_monitor import done
from experiments.babilong_linked_selection.long_rollout_monitors import unit_contents


def test_local_monitors_are_hourly_and_reuse_the_existing_workspace():
    workspace = Path("/scratch/cross")
    root = workspace / "query_cartridges_linked_roll32_dev_20261104"
    for kind in ("submit", "dev", "holdout"):
        prefix, service, timer = unit_contents(workspace, root, kind)
        assert prefix.endswith(kind)
        assert "--workspace /scratch/cross" in service
        assert "OnActiveSec=1h" in timer and "OnUnitActiveSec=1h" in timer
        assert "sbatch" not in service
    with pytest.raises(ValueError): unit_contents(workspace, root, "invalid")


def test_complete_report_must_not_hide_an_incomplete_pairing():
    assert done({"models": {"qwen": {"decision": {"reason": "beats_all_development_controls"}}}})
    assert not done({"models": {"qwen": {"decision": {"reason": "development_incomplete"}}}})
    assert not done({"models": {}})
