import json
import sys
from types import SimpleNamespace

import experiments.value_lr_physical.monitor as monitor
from test_value_lr_physical_report import fixture_report


def test_monitor_does_not_treat_pending_or_unobservable_as_terminal(tmp_path, monkeypatch):
    root = tmp_path / "query_cartridges_value_lr_physical_gpt_holdout_20261104"
    root.mkdir()
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "cartridge_job_watch", SimpleNamespace(record_terminal=lambda _root: None))
    assert monitor.advance(tmp_path, root) == {"stage": "in_progress", "scheduler_state": None}


def test_completed_holdout_combines_with_frozen_200_case_development(tmp_path, monkeypatch):
    development, report, manifest, holdout = fixture_report()
    root = tmp_path / "query_cartridges_value_lr_physical_gpt_holdout_20261104"
    root.mkdir()
    for name, data in (("development_manifest.json", development), ("development_snapshot.json", report),
                       ("manifest.json", manifest), ("latest.json", holdout), ("finished.json", {})):
        (root / name).write_text(json.dumps(data))
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_args, **_kwargs: None)
    result = monitor.advance(tmp_path, root)
    assert result["stage"] == "complete"
    combined = json.loads((root / "combined_1000.json").read_text())
    assert combined["overall"]["n"] == 1000 and combined["fixed_value_learning_rate"] == .001
