from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from experiments.rollout_value_pair import job, monitor, report
from test_study_job import source_protocols, sources


def test_manifest_and_batch_fix_two_original_sd_settings():
    manifest = job.make_manifest(*source_protocols())
    assert manifest["models"] == ["qwen"] and manifest["case_count"] == 200
    assert (manifest["baseline_rollout_tokens"], manifest["baseline_value_lr"]) == (8, .0001)
    assert (manifest["student_rollout_tokens"], manifest["candidate_value_lr"]) == (32, .001)
    assert manifest["online_api_calls"] == 0 and manifest["reference_answers_used_in_inference"] is False
    for name in (job.DEV, job.HOLDOUT):
        script = job.batch_script(Path("/tmp") / name)
        assert "#SBATCH --gpus-per-task=1" in script and "#SBATCH --mem=64G" in script
        assert "rollout_value_pair/execute.py" in script and "unset OPENAI_API_KEY" in script
        assert 'TRITON_CACHE_DIR="$TASK_ROOT/runtime_cache/triton"' in script
        assert 'TORCH_EXTENSIONS_DIR="$TASK_ROOT/runtime_cache/torch_extensions"' in script
    bad = list(source_protocols())
    bad[2]["gain_learning_rate"] = .1
    with pytest.raises(ValueError, match="baseline changed"):
        job.make_manifest(*bad)


def test_submit_uses_one_gpu_slot_and_is_idempotent(tmp_path, monkeypatch):
    root = tmp_path / job.DEV
    root.mkdir()
    (root / "cpu_validation_passed.json").write_text(json.dumps(dict(cpu_tests_passed=True)))
    (root / "job.sbatch").write_text("# test\n")
    @contextmanager
    def available():
        yield True
    monkeypatch.setitem(sys.modules, "gpu_slot_budget", SimpleNamespace(submission_slot=available))
    calls = []
    monkeypatch.setattr(job.subprocess, "run", lambda args, **_kwargs:
                        calls.append(args) or SimpleNamespace(stdout="123\n"))
    sent = job.submit(tmp_path, root)
    assert sent["job_id"] == "123" and sent["gpus"] == 1 and sent["cases"] == 200
    assert job.submit(tmp_path, root)["job_id"] == "123" and len(calls) == 1


def test_holdout_requires_development_outperformance(tmp_path):
    root = tmp_path / job.HOLDOUT
    root.mkdir()
    stats = dict(n=200, accuracy=dict(whole=60, zero=60, teacher=70, cartridge=65),
                 strictly_paired=dict(n=200, delta_pp=1, new_correct=3, new_wrong=1))
    dev = dict(mode="development", expected=200, metric="answer_focused_lexical_v2",
               errors=[], invalid=[], overall=stats)
    (root / "development_snapshot.json").write_text(json.dumps(dev))
    with pytest.raises(ValueError, match="did not beat"):
        job.submit(tmp_path, root)


def test_combined_report_requires_source_disjoint_complete_cases():
    dev, held = sources()
    def rows(cases):
        return [dict(case=case, task="qa1", length=8192, whole=0, zero=0, teacher=0,
            baseline=0, cartridge=1, cartridge_percent=10.) for case in cases]
    manifest = dict(development_case_ids=dev, holdout_case_ids=held,
                    qualitative_case_sources=["qa1-300023", "qa2-300037"])
    base = dict(metric="answer_focused_lexical_v2", errors=[], invalid=[])
    data = monitor.combine(manifest, dict(base, rows=rows(dev)), dict(base, rows=rows(held)))
    assert data["overall"]["n"] == 1000
    assert data["holdout_excluding_two_previously_viewed_sources"]["n"] == 792
    with pytest.raises(ValueError, match="Complete"):
        monitor.combine(manifest, dict(base, rows=rows(dev)), dict(base, rows=rows(held)[:-1]))


def test_report_rejects_nonmatching_hyperparameters_and_score_before_validation(monkeypatch):
    called = []
    monkeypatch.setattr(report, "validate_new", lambda *_: called.append(True))
    p = dict(baseline_training_steps=2, candidate_training_steps=2, baseline_verified=0,
        baseline_rollout_tokens=8, baseline_value_lr=.0001,
        candidate_verified=0, baseline_native_prediction="A", reference_answers_used=False,
        teacher_probability_tensors_compared=False, changed_settings=["student_rollout_tokens", "value_learning_rate"],
        same_loaded_model=True, same_gpu=True, same_request=True, same_practice_checks=True,
        independently_prefilled_caches=True, same_selection=True, same_teacher_tokens=True, same_zero_tokens=True,
        baseline_standalone_released=True, candidate_standalone_released=True)
    training = dict(gain_head_mode="kv", gain_learning_rate=.01, value_learning_rate=.001,
        adaptive_max_answer_steps=8, student_question_matches_teacher=False, steps=[{}, {}])
    result = dict(rollout_value_pair=p, training=training, self_study=dict(verified=0),
                  self_judge=dict(native_prediction="A"))
    report.validate_pair(result, "case", "qa3")
    assert called == [True]
    result["training"]["value_learning_rate"] = .0001
    with pytest.raises(ValueError, match="hyperparameter pair"):
        report.validate_pair(result, "case", "qa3")


def test_partial_report_scores_only_existing_validated_outputs(tmp_path, monkeypatch):
    manifest = job.make_manifest(*source_protocols())
    case = manifest["case_ids"][0]
    folder = tmp_path / "cases/qwen" / case
    folder.mkdir(parents=True)
    result = dict(length=8192, self_judge=dict(native_prediction="wrong"),
        teacher=dict(prediction="right"), prediction="right",
        rollout_value_pair=dict(baseline_prediction="wrong"),
        standalone_cartridge=dict(original_question_controls=dict(zero=dict(prediction="wrong")),
                                  memory=dict(standalone_cartridge_bytes=100)),
        memory=dict(full_document_kv_bytes=1000))
    (folder / "result.json").write_text(json.dumps(result))
    seen = []
    monkeypatch.setattr(report, "validate_pair", lambda item, actual_case, task: seen.append((actual_case, task)))
    references = {case: dict(task="qa1", answers=["right"])}
    score = lambda text, answers, *_: (text in answers, None)
    data = report.collect(manifest, tmp_path, references, score)
    assert seen == [(case, "qa1")] and data["overall"]["n"] == 1
    assert data["overall"]["strictly_paired"]["new_correct"] == 1
    assert data["overall"]["accuracy"]["teacher"] == 100
    assert not data["errors"] and not data["invalid"]
