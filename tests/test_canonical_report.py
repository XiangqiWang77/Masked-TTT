import pytest

from experiments.value_lr_physical.canonical_report import development_decision, render, summarize


def row(*, whole=0, original_zero=0, original_teacher=0, original_cartridge=0,
        canonical_zero=0, canonical_teacher=0, canonical_cartridge=0):
    return dict(whole=whole, original_zero=original_zero, original_teacher=original_teacher,
        original_cartridge=original_cartridge, canonical_zero=canonical_zero,
        canonical_teacher=canonical_teacher, canonical_cartridge=canonical_cartridge,
        same_teacher_tokens=True, same_zero_tokens=True, same_practice_checks=False,
        teacher_prompt_length_changed=True)


def report(rows):
    return dict(mode="development", expected_per_model=200, metric="answer_focused_lexical_v2",
                errors=[], invalid=[], models={"qwen": dict(overall=summarize(rows))})


def test_paired_comparisons_measure_gains_and_regressions_separately():
    rows = [row(canonical_cartridge=1), row(original_cartridge=1, canonical_teacher=1),
            row(original_zero=1, canonical_zero=1, canonical_cartridge=1), row()]
    result = summarize(rows)
    assert result["versus"]["original_cartridge"] == dict(new_correct=2, new_wrong=1, delta_pp=25)
    assert result["versus"]["canonical_zero"] == dict(new_correct=1, new_wrong=0, delta_pp=25)
    assert result["practice_checks_changed"] == 4 and result["teacher_prompt_length_changed"] == 4
    assert summarize([])["versus"]["whole"]["delta_pp"] is None
    rows[0]["canonical_cartridge"] = True
    with pytest.raises(ValueError, match="binary"):
        summarize(rows)


def test_full_development_must_beat_all_own_controls_before_promotion():
    rows = [row(canonical_cartridge=1)] * 101 + [row()] * 99
    candidate = report(rows)
    assert development_decision(candidate, "qwen")["eligible"]
    candidate["errors"].append(dict(model="qwen"))
    assert development_decision(candidate, "qwen")["reason"] == "development_incomplete_or_invalid"
    candidate = report(rows[:-1])
    assert not development_decision(candidate, "qwen")["eligible"]
    candidate = report([row(canonical_teacher=1, canonical_cartridge=1)] * 200)
    assert development_decision(candidate, "qwen")["reason"] == "does_not_beat_same_run_controls"


def test_table_columns_match_the_emitted_arm_order():
    candidate = report([row(original_zero=1, original_teacher=0, canonical_zero=0, canonical_teacher=1)])
    table = render(candidate)
    assert "Original zero | Original Teacher" in table and "Canonical zero | Canonical Teacher" in table
    assert "| qwen | 1 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 |" in table
