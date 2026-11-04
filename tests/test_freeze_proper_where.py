import pytest

from experiments.babilong_linked_selection.freeze_proper_where import summarize


def sample_report():
    rows = []
    for i in range(200):
        changed = i < 7
        row = dict(case=f"case-{i % 4}-{i // 4}", task="qa1", length=8192,
                   selection_changed=changed, same_whole_output=True,
                   same_teacher_tokens=True, same_zero_tokens=True,
                   same_cartridge_tokens=True)
        row.update({k: 0 for k in ("whole", "zero", "teacher", "cartridge", "old_whole", "old_zero", "old_teacher", "old_cartridge")})
        if i in (0, 4, 8, 12):
            row["cartridge"] = row["teacher"] = 1
        rows.append(row)
    return dict(mode="development", model="qwen", metric="answer_focused_lexical_v2",
                expected=200, online_api_calls=0, reference_answers_used_only_posthoc=True,
                errors=[], invalid=[], unmatched=[], rows=rows,
                expected_development_changed_selections=7)


def test_aggregate_distinguishes_changed_and_unchanged_sources():
    result = summarize(sample_report())
    assert result["overall"]["cases"] == 200
    assert result["overall"]["source_questions"] == 50
    assert result["changed"]["cases"] == 7
    assert result["changed"]["cartridge_vs_old_cartridge"]["new_correct"] == 2
    assert result["unchanged"]["cartridge_vs_old_cartridge"]["new_correct"] == 2
    assert "case-0-0" not in str(result)


def test_rejects_incomplete_and_unexpected_protocol():
    report = sample_report()
    report["rows"].pop()
    with pytest.raises(ValueError):
        summarize(report)
    report = sample_report()
    report["reference_answers_used_only_posthoc"] = False
    with pytest.raises(ValueError):
        summarize(report)
