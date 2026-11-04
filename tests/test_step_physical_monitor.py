import json
import sys
from types import SimpleNamespace

import experiments.step_physical.monitor as monitor
from experiments.step_physical.protocol import holdout_manifest
from experiments.step_physical.report import result
from test_step_physical import fixture_development


def setup(tmp_path):
    development, report, profile = fixture_development()
    manifest = holdout_manifest(development, report, profile, "gpt")
    root = tmp_path / "query_cartridges_step_physical_gpt_holdout_20261104"
    root.mkdir()
    (root / "report.py").write_text("# synthetic report\n")
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "development_manifest.json").write_text(json.dumps(development))
    (root / "development_snapshot.json").write_text(json.dumps(report))
    return root, manifest


def test_partial_monitor_reports_recovery_without_counting_missing_cases(tmp_path, monkeypatch):
    root, manifest = setup(tmp_path)
    case = manifest["case_ids"][0]
    row = dict(case=case, task="qa1", length=8192, whole=1, zero=0, teacher=0,
               step2=0, step4=0, step8=1, cartridge=1, cartridge_percent=10.)
    (root / "latest.json").write_text(json.dumps(result("holdout", manifest, [row])))
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "cartridge_job_watch", SimpleNamespace(record_terminal=lambda _root: None))
    status = monitor.advance(tmp_path, root)
    assert status["stage"] == "in_progress" and status["cases"] == 1
    recovery = json.loads((root / "recovery_latest.json").read_text())
    assert recovery["overall"]["n"] == 1 and recovery["overall"]["whole_only"]["recovered"] == 1


def test_completed_monitor_combines_fixed_development_and_holdout(tmp_path, monkeypatch):
    root, manifest = setup(tmp_path)
    rows = [dict(case=case, task=case.split("-", 2)[2].split("-", 1)[0], length=8192,
                 whole=int(i < 100), zero=int(i < 100), teacher=int(i < 100),
                 step2=int(i < 100), step4=int(i < 110), step8=int(i < 140),
                 cartridge=int(i < 140), cartridge_percent=10.)
            for i, case in enumerate(manifest["case_ids"])]
    (root / "latest.json").write_text(json.dumps(result("holdout", manifest, rows)))
    (root / "finished.json").write_text("{}")
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_args, **_kwargs: None)
    status = monitor.advance(tmp_path, root)
    assert status["stage"] == "complete"
    combined = json.loads((root / "combined_1000.json").read_text())
    assert combined["overall"]["n"] == 1000 and combined["holdout"]["n"] == 800
    assert json.loads((root / "recovery_latest.json").read_text())["overall"]["n"] == 1000
