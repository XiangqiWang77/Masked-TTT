import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.self_study_prompt.execute import deferred_probe_action
from experiments.self_study_prompt.inline_probe import ProbeDeferred, ProbeFailed, ProbeRejected, run_inline_probe


def setup_probe(tmp_path, *, complete=True, improved=True, errors=None):
    root = tmp_path
    (root / "manifest.json").write_text(json.dumps(dict(models=["qwen"], mode="development",
        probe_gate_mode="in_job_before_dataset", reference_answers_used_in_inference=False, online_api_calls=0)))
    directory = root / "study_probe"
    directory.mkdir()
    (directory / "driver.py").write_text("def run_if_needed(engine, model, identity):\n    return 'already_complete'\n")
    cases = ["case-" + str(i) for i in range(10)]
    def info(verified):
        return dict(generated_tokens=4, parsed=verified, verified=verified, rejections={},
                    reference_answers_used=False, student_answer_hint_used=False)
    rows = {key: dict(base=info(0), variant=info(int(improved))) for key in (cases if complete else cases[:4])}
    data = dict(model="qwen", cases=cases, rows=rows, errors=errors or [], complete=complete,
                reference_answers_used=False, online_api_calls=0, training_runtime_changed=False)
    (directory / "qwen.json").write_text(json.dumps(data))
    return root


def test_probe_accepts_only_more_verified_source_exercises(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "test-device")
    root = setup_probe(tmp_path)
    result = run_inline_probe(SimpleNamespace(device="cuda:0"), root)
    assert result["source_probe_complete"] and result["source_probe_more_verified"]
    assert (root / "probe_verified.json").is_file()


def test_probe_rejects_ties_and_defers_incomplete_without_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "test-device")
    # Create separate roots explicitly; no real scheduler or GPU is consulted.
    rejected_root = tmp_path / "rejected"; rejected_root.mkdir()
    incomplete_root = tmp_path / "incomplete"; incomplete_root.mkdir()
    failed_root = tmp_path / "failed"; failed_root.mkdir()
    with pytest.raises(ProbeRejected):
        run_inline_probe(SimpleNamespace(device="cuda:0"), setup_probe(rejected_root, improved=False))
    with pytest.raises(ProbeDeferred) as deferred:
        run_inline_probe(SimpleNamespace(device="cuda:0"), setup_probe(incomplete_root, complete=False))
    assert deferred.value.progress["n"] == 4
    with pytest.raises(ProbeFailed):
        run_inline_probe(SimpleNamespace(device="cuda:0"), setup_probe(failed_root, complete=False,
                         errors=[dict(case="case-0", error_type="ValueError")]))


def test_deferred_retry_is_bounded_and_accounts_for_active_time(tmp_path):
    budget = dict(spent_seconds=100., total_budget_seconds=5000.,
                  active=dict(remaining_seconds=1000., started_wall=1000.))
    (tmp_path / "allocation_budget.json").write_text(json.dumps(budget))
    action = deferred_probe_action(tmp_path, dict(n=4, expected=10, errors=0), now=1100.)
    assert action["requeue"] and action["remaining_budget_seconds"] == 4800.
    (tmp_path / "probe_deferred.json").write_text(json.dumps(action))
    action = deferred_probe_action(tmp_path, dict(n=4, expected=10, errors=0), now=1100.)
    assert action["requeue"] and action["stalls"] == 1
    (tmp_path / "probe_deferred.json").write_text(json.dumps(action))
    action = deferred_probe_action(tmp_path, dict(n=4, expected=10, errors=0), now=1100.)
    assert not action["requeue"] and action["stalls"] == 2
