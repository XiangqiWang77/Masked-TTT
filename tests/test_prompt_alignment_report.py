from copy import deepcopy
import json
import pytest
from experiments.prompt_alignment import report


def test_summary_counts_both_directions():
    rows = [dict(case="babilong-8-qa3-1", whole=0, zero=0, teacher=0, baseline=0, aligned=1),
            dict(case="babilong-16-qa3-1", whole=1, zero=1, teacher=1, baseline=1, aligned=0)]
    result = report.summarize(rows)
    assert result["n"] == 2 and result["source_questions"] == 1
    assert result["aligned_vs_baseline"]["new_correct"] == 1
    assert result["aligned_vs_baseline"]["new_wrong"] == 1
    assert result["aligned_vs_baseline"]["delta_pp"] == 0


def test_pair_rejects_mismatched_evidence_teacher_and_control(monkeypatch):
    monkeypatch.setattr(report, "validate_new", lambda *args: None)
    monkeypatch.setattr(report, "validate_steps", lambda *args: None)
    selection = dict(selected_original_chunks=[[1]], selected_chunks=[1], total_chunks=10,
                     selected_tokens=100)
    cartridge = dict(original_question_controls=dict(zero=dict(tokens=[1])), answer_format="original")
    candidate = dict(selection=selection, teacher=dict(tokens=[2]),
        self_study=dict(verified=0, checks=[]), standalone_cartridge=cartridge,
        training=dict(student_question_matches_teacher=True, gain_head_mode="kv", steps=[{}, {}]))
    baseline = deepcopy(candidate)
    report.validate_pair(candidate, baseline, "case", "qwen", "qa1")
    for key, value, message in (("selection", dict(selection, selected_chunks=[2]), "chunks"),
                                ("teacher", dict(tokens=[7]), "Teacher"),
                                ("self_study", dict(verified=0, checks=[{}]), "Practice"),
                                ("standalone_cartridge", dict(original_question_controls=
                                    dict(zero=dict(tokens=[7])), answer_format="original"), "zero")):
        altered = deepcopy(baseline)
        altered[key] = value
        with pytest.raises(ValueError, match=message):
            report.validate_pair(candidate, altered, "case", "qwen", "qa1")


def test_unmatched_historical_baseline_keeps_same_run_candidate_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "validate_partition", lambda *_: None)
    monkeypatch.setattr(report, "validate_new", lambda *_: None)
    monkeypatch.setattr(report, "validate_steps", lambda *_: None)
    cases = ["babilong-8-qa1-" + str(i) for i in range(200)]
    manifest = dict(development_case_ids=cases, holdout_case_ids=["held"], mode="development",
        case_count=200, case_ids=cases, selection_policy="decision_entity_link_v1",
        entity_reserve_fraction=.25, keep_fraction=.1, student_rollout_tokens=32,
        selected_transfer_weight=0, align_student_teacher_prompt=True,
        reference_answers_used_in_inference=False, online_api_calls=0, models=["qwen"])
    selected = dict(selected_original_chunks=[[1]], selected_chunks=[1], total_chunks=10, selected_tokens=1)
    controls = dict(zero=dict(tokens=[1], prediction="good"))
    candidate = dict(key=cases[0], length=8192, selection=selected, teacher=dict(tokens=[2], prediction="good"),
        self_study=dict(verified=0, checks=[]), self_judge=dict(native_prediction="bad"), prediction="good",
        training=dict(student_question_matches_teacher=True, gain_head_mode="kv", steps=[{}, {}]),
        standalone_cartridge=dict(original_question_controls=controls, answer_format="original"))
    baseline = deepcopy(candidate)
    baseline["teacher"]["tokens"] = [99]
    for root, result in ((tmp_path / "new", candidate), (tmp_path / "old", baseline)):
        path = root / "cases/qwen" / cases[0] / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result))
    refs = {cases[0]: dict(task="qa1", answers=["good"])}
    score = lambda text, *_args: (int(text == "good"), True)
    result = report.collect(manifest, tmp_path / "new", tmp_path / "old", refs, score)
    assert result["models"]["qwen"]["candidate_all"]["n"] == 1
    assert result["models"]["qwen"]["overall"]["n"] == 0
    assert not result["invalid"] and len(result["unmatched_baselines"]) == 1
