import json
from pathlib import Path

from experiments.self_study_prompt import study_job, study_monitor, study_trigger


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


def test_trigger_waits_without_probe_and_submits_only_after_positive(tmp_path, monkeypatch):
    workspace, root = tmp_path, tmp_path / study_job.DEV
    root.mkdir()
    calls = []
    monkeypatch.setattr(study_trigger, "submit", lambda *_: calls.append("submit") or dict(job_id="123"))
    monkeypatch.setattr(study_trigger, "enable_monitor", lambda *_: calls.append("monitor"))
    assert study_trigger.advance(workspace, root, Path("/python"))["state"] == "waiting_for_prompt_probe"
    probe = workspace / study_job.PROBE
    probe.parent.mkdir(parents=True)
    report = dict(state="complete", n=10, expected=10, errors=0, reference_answers_used=False,
        online_api_calls=0, training_runtime_changed=False, same_selected_teacher_cache=True,
        downstream_accuracy_not_measured=True, base=dict(n=10, verified_exercises=1),
        variant=dict(n=10, verified_exercises=1), more_verified=0, fewer_verified=0)
    probe.write_text(json.dumps(report))
    assert study_trigger.advance(workspace, root, Path("/python"))["state"] == "probe_not_eligible"
    assert not calls
    report.update(variant=dict(n=10, verified_exercises=3), more_verified=2)
    probe.write_text(json.dumps(report))
    assert study_trigger.advance(workspace, root, Path("/python"))["state"] == "submitted"
    assert calls == ["submit", "monitor"]
    assert study_trigger.advance(workspace, root, Path("/python"))["state"] == "submitted"
    assert calls == ["submit", "monitor", "submit"]  # Do not restart a completed monitor.


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
