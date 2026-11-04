from copy import deepcopy
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
