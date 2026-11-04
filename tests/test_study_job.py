import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from experiments.self_study_prompt import study_job, study_monitor


def sources():
    dev = [f"qa1-{900000+i}" for i in range(50)]
    held = ["qa1-300023", "qa2-300037"] + [f"qa2-{800000+i}" for i in range(198)]
    return ([f"babilong-{length}-{source}" for source in dev for length in ("8", "16", "32", "64")],
            [f"babilong-{length}-{source}" for source in held for length in ("8", "16", "32", "64")])


def source_protocols():
    dev, held = sources()
    common = dict(case_ids=dev, holdout_case_ids=held, model_protocols={"qwen": "protocol"},
                  keep_fraction=.1, reference_answers_used_in_inference=False, online_api_calls=0,
                  qualitative_case_sources=["qa1-300023", "qa2-300037"])
    rollout = dict(common, selection_policy="cached_decision_score10", student_rollout_tokens=32)
    linked = dict(common, selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25)
    profile = dict(student_rollout_tokens=8, steps=2, gain_learning_rate=.01, value_learning_rate=.0001,
                   self_question_count=2, max_answer_steps=8, answer_KL_stop=.5, guard_enabled=True)
    return rollout, linked, profile


def test_freezes_olmo_conditions_and_one_gpu():
    manifest = study_job.make_manifest(*source_protocols())
    assert manifest["models"] == ["qwen"] and manifest["case_count"] == 200
    assert manifest["self_study_prompt_comparison"] == "original_vs_study_only"
    assert manifest["online_api_calls"] == 0 and manifest["reference_answers_used_in_inference"] is False
    for name in (study_job.DEV, study_job.HOLDOUT):
        script = study_job.batch_script(Path("/tmp") / name)
        assert "#SBATCH --gpus-per-task=1" in script and "#SBATCH --mem=64G" in script
        assert "self_study_prompt/execute.py" in script and "unset OPENAI_API_KEY" in script


def test_submit_requires_in_job_probe_gate_and_one_gpu_slot(tmp_path, monkeypatch):
    workspace, root = tmp_path, tmp_path / study_job.DEV
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(dict(probe_gate_mode="in_job_before_dataset")))
    (root / "cpu_validation_passed.json").write_text(json.dumps(dict(cpu_tests_passed=True)))
    with pytest.raises(ValueError, match="in-job prompt gate"):
        study_job.submit(workspace, root)
    (root / "study_probe").mkdir()
    (root / "study_probe/driver.py").write_text("# gated probe\n")
    (root / "job.sbatch").write_text("# test\n")
    @contextmanager
    def available():
        yield True
    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=available))
    calls = []
    monkeypatch.setattr(study_job.subprocess, "run", lambda args, **_kwargs:
                        calls.append(args) or SimpleNamespace(stdout="123\n"))
    result = study_job.submit(workspace, root)
    assert result["job_id"] == "123" and result["gpus"] == 1 and result["cases"] == 200
    assert len(calls) == 1 and calls[0][0] == "sbatch"
    assert study_job.submit(workspace, root)["job_id"] == "123" and len(calls) == 1


def test_combined_report_keeps_previously_inspected_sources_separate():
    dev, held = sources()
    def rows(cases):
        return [dict(case=case, task="qa1", length=8192, whole=0, zero=0, teacher=0,
            baseline=0, cartridge=1, cartridge_percent=10., baseline_verified=0,
            candidate_verified=1, same_practice_checks=False) for case in cases]
    development, holdout = rows(dev), rows(held)
    manifest = dict(development_case_ids=dev, holdout_case_ids=held,
                    qualitative_case_sources=["qa1-300023", "qa2-300037"])
    base = dict(metric="answer_focused_lexical_v2", errors=[], invalid=[])
    report = study_monitor.combine(manifest, dict(base, rows=development), dict(base, rows=holdout))
    assert report["overall"]["n"] == 1000
    assert report["holdout_excluding_two_previously_viewed_sources"]["n"] == 792
    assert report["practice"]["more"] == 1000
