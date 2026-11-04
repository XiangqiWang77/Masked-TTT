"""A completed pilot can be reused without touching the source-disjoint holdout."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from experiments.long_rollout_steps.extend import _copy_atomic, extension_manifest, submit
from experiments.long_rollout_steps.prepare import select_pilot


def fixture_protocol():
    dev_sources = [f"qa{task}-{400000 + task * 100 + source}"
                   for task in range(1, 11) for source in range(5)]
    hold_sources = [f"qa{task}-{500000 + task * 100 + source}"
                    for task in range(1, 11) for source in range(20)]
    hold_sources[0], hold_sources[20] = "qa1-300023", "qa2-300037"
    expand = lambda sources: [f"babilong-{length}-{source}" for source in sources
                              for length in (8, 16, 32, 64)]
    development, holdout = expand(dev_sources), expand(hold_sources)
    pilot_cases = select_pilot(development)
    pilot = dict(case_count=56, case_ids=pilot_cases, development_case_ids=development,
        holdout_case_ids=holdout, step_budgets=[2, 4, 8], student_rollout_tokens=32,
        selected_transfer_weight=0., selection_policy="decision_entity_link_v1", keep_fraction=.1,
        reference_answers_used_in_inference=False, online_api_calls=0, models=["qwen", "gpt"])
    report = dict(expected_per_model=56, metric="answer_focused_lexical_v2", cached_scores=True,
        online_api_calls=0, evaluation_answers_used_only_posthoc=True, errors=[], invalid=[],
        models={model: {"rows": [{"case": case} for case in pilot_cases]} for model in pilot["models"]})
    return pilot, report


def test_expand_preserves_all_56_without_using_holdout():
    pilot, report = fixture_protocol()
    expanded = extension_manifest(pilot, report)
    assert expanded["case_count"] == 200
    assert expanded["carried_forward_cases_per_model"] == 56
    assert expanded["new_cases_per_model"] == 144
    assert set(pilot["case_ids"]).issubset(expanded["case_ids"])
    assert not set(expanded["case_ids"]) & set(expanded["holdout_case_ids"])
    assert not expanded["configuration_selected_from_pilot"]
    assert not expanded["holdout_evaluated_at_selection"]


@pytest.mark.parametrize("defect", ["missing_case", "duplicate_case", "invalid_result", "missing_model"])
def test_partial_or_invalid_pilot_cannot_be_promoted(defect):
    pilot, report = fixture_protocol()
    if defect == "missing_case":
        report["models"]["qwen"]["rows"].pop()
    elif defect == "duplicate_case":
        report["models"]["qwen"]["rows"][-1] = report["models"]["qwen"]["rows"][0]
    elif defect == "invalid_result":
        report["invalid"].append({"case": pilot["case_ids"][0]})
    else:
        del report["models"]["gpt"]
    with pytest.raises(ValueError):
        extension_manifest(pilot, report)


def test_copy_is_atomic_idempotent_and_rejects_conflicting_result(tmp_path):
    source, target = tmp_path / "source.json", tmp_path / "cases/a/result.json"
    source.write_text('{"answer": "blue"}\n')
    _copy_atomic(source, target)
    _copy_atomic(source, target)
    assert target.read_bytes() == source.read_bytes()
    source.write_text('{"answer": "red"}\n')
    with pytest.raises(ValueError, match="differs"):
        _copy_atomic(source, target)
    assert target.read_text() == '{"answer": "blue"}\n'


def test_submit_is_one_gpu_guarded_and_idempotent(tmp_path, monkeypatch):
    import experiments.long_rollout_steps.extend as implementation
    pilot, report = fixture_protocol()
    root = tmp_path / "query_cartridges_longsteps_200dev_20261104"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(extension_manifest(pilot, report)))
    (root / "prepare_complete.json").write_text("{}")
    (root / "job.sbatch").write_text("test")
    @contextmanager
    def slot():
        yield True
    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=slot))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345\n")
    monkeypatch.setattr(implementation.subprocess, "run", run)
    result = submit(tmp_path, root)
    assert result["job_id"] == "12345" and result["gpus"] == 1
    assert result["cases_per_model"] == 200 and result["cases_reused_per_model"] == 56
    assert submit(tmp_path, root) == result and len(calls) == 1
    assert "--job-name=qc-longsteps-200" in calls[0]


def test_submit_respects_shared_gpu_cap(tmp_path, monkeypatch):
    root = tmp_path / "query_cartridges_longsteps_200dev_20261104"
    root.mkdir()
    pilot, report = fixture_protocol()
    (root / "manifest.json").write_text(json.dumps(extension_manifest(pilot, report)))
    (root / "prepare_complete.json").write_text("{}")
    @contextmanager
    def full():
        yield False
    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=full))
    with pytest.raises(RuntimeError, match="12-GPU"):
        submit(tmp_path, root)
    assert not (root / "submission.json").exists()
