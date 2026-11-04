import pytest

from experiments.value_lr_physical.audit_practice_signal import summarize


def row(teacher, selected, cartridge, matched_zero, matched_trained, changed):
    return dict(verified=0, first_kl=0.2, gain_grad=0.01, value_grad=0.03, steps=2,
                whole=1, teacher=teacher, selective=selected, cartridge=cartridge,
                selective_teacher_format=matched_zero, cartridge_teacher_format=matched_trained,
                original_tokens_changed=changed, teacher_format_tokens_changed=changed)


def test_paired_gains_and_regressions_use_same_question_and_suffix():
    data = [row(0, 0, 1, 0, 1, True), row(1, 1, 0, 1, 1, True),
            row(1, 0, 1, 0, 0, False), row(0, 0, 0, 0, 0, False)]
    got = summarize(data)
    assert got["teacher_accuracy"] == 50 and got["cartridge_accuracy"] == 50
    assert got["vs_teacher"] == dict(new_correct=1, new_wrong=1, delta_points=0)
    assert got["vs_selective"] == dict(new_correct=2, new_wrong=1, delta_points=25)
    assert got["vs_selective_teacher_format"] == dict(new_correct=1, new_wrong=0, delta_points=25)
    assert got["original_tokens_changed"] == 2


def test_partial_or_nonbinary_outcomes_are_rejected():
    data = [row(0, 0, 1, 0, 1, True), row(1, 1, 0, 1, 0, False)]
    data[0]["teacher"] = None
    with pytest.raises(ValueError, match="missing"):
        summarize(data)
    data[0]["teacher"] = 2
    with pytest.raises(ValueError, match="binary"):
        summarize(data)
