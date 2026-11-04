import pytest
import json
import sys
from types import SimpleNamespace
from experiments.prompt_alignment import holdout_report, promote


def example_manifest():
    development = ["babilong-8-qa1-" + str(i) for i in range(200)]
    holdout = ["babilong-8-qa2-" + str(i) for i in range(800)]
    return dict(mode="development", case_count=200, case_ids=development,
        development_case_ids=development, holdout_case_ids=holdout,
        align_student_teacher_prompt=True, selected_transfer_weight=0,
        reference_answers_used_in_inference=False, online_api_calls=0, models=["qwen", "gpt"])


def report_for(manifest):
    def item(accuracy):
        return dict(candidate_completed=200, overall=dict(n=200, accuracy=accuracy),
                    rows=[dict(case=case) for case in manifest["case_ids"]])
    return dict(metric="answer_focused_lexical_v2", expected_per_model=200,
        models=dict(qwen=item(dict(whole=50, zero=57, teacher=60, baseline=59, aligned=61)),
                    gpt=item(dict(whole=50, zero=57, teacher=60, baseline=59, aligned=59))),
        errors=[], invalid=[])


def test_only_strictly_better_complete_model_is_promoted(monkeypatch):
    monkeypatch.setattr(promote, "validate_partition", lambda *args: None)
    manifest = example_manifest(); report = report_for(manifest)
    models, decisions = promote.choose_models(manifest, report)
    assert models == ["qwen"] and decisions["qwen"]["eligible"]
    assert not decisions["gpt"]["eligible"]
    held = promote.holdout_manifest(manifest, models, decisions)
    assert held["mode"] == "holdout" and held["models"] == ["qwen"]
    assert held["case_ids"] == manifest["holdout_case_ids"]
    assert held["holdout_used_for_configuration_selection"] is False
    report["invalid"].append(dict(model="qwen"))
    models, decisions = promote.choose_models(manifest, report)
    assert not models and decisions["qwen"]["reason"] == "incomplete_or_unmatched"


def test_summary_and_combined_cases_are_checked():
    manifest = example_manifest()
    def rows(cases, trained):
        return [dict(case=case, task="qa1", length=8, whole=0, zero=0,
                     teacher=0, aligned=trained) for case in cases]
    development = dict(metric="answer_focused_lexical_v2",
        models=dict(qwen=dict(rows=rows(manifest["development_case_ids"], 1))))
    holdout = dict(metric="answer_focused_lexical_v2", errors=[], invalid=[],
        models=dict(qwen=dict(rows=rows(manifest["holdout_case_ids"], 1))))
    manifest["models"] = ["qwen"]
    combined = holdout_report.combine(manifest, development, holdout)
    assert combined["models"]["qwen"]["overall"]["n"] == 1000
    assert combined["models"]["qwen"]["holdout"]["n"] == 800
    assert combined["models"]["qwen"]["overall"]["aligned_vs_teacher"]["delta_pp"] == 100
    holdout["models"]["qwen"]["rows"].pop()
    assert holdout_report.combine(manifest, development, holdout) is None
    holdout["errors"] = [dict(model="qwen")]
    with pytest.raises(ValueError, match="not valid"):
        holdout_report.combine(manifest, development, holdout)


def test_single_gpu_holdout_script_and_summary():
    from pathlib import Path
    script = promote.holdout_script(Path("/tmp/query_cartridges_prompt_aligned_holdout_20261104"))
    assert "#SBATCH --gpus-per-task=1" in script
    assert "prompt_alignment/execute.py" in script
    rows = [dict(case="babilong-8-qa1-1", whole=0, zero=0, teacher=0, aligned=1),
            dict(case="babilong-16-qa1-1", whole=1, zero=1, teacher=1, aligned=0)]
    result = holdout_report.summarize(rows)
    assert result["aligned_vs_teacher"]["new_correct"] == 1
    assert result["aligned_vs_teacher"]["new_wrong"] == 1


def test_promotion_distinguishes_waiting_and_terminal_prerequisite(tmp_path, monkeypatch):
    development = tmp_path / "query_cartridges_prompt_aligned_dev_20261104"
    baseline = tmp_path / "query_cartridges_longsteps_200dev_20261104"
    development.mkdir(); baseline.mkdir()
    (development / "submission.json").write_text(json.dumps(dict(job_id="123")))
    (baseline / "submission.json").write_text(json.dumps(dict(job_id="456")))
    states = {"123": None, "456": None}
    monkeypatch.setitem(sys.modules, "cartridge_job_watch", SimpleNamespace(observed_terminal=states.get))
    assert promote.freeze(tmp_path) == dict(state="waiting")
    states["456"] = "FAILED"
    assert promote.freeze(tmp_path) == dict(state="prerequisite_terminal", prerequisite="baseline",
                                               job_id="456", scheduler_state="FAILED")
