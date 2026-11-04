"""Check paired accounting, including the already-correct selected cases."""
import pytest

from experiments.babilong_linked_selection.audit_original_transfer import summarize


def row(case, whole, adapted, selected, teacher, cartridge):
    return dict(case=f"babilong-8-qa1-{case}", whole=whole, whole_adapted=adapted,
                selective=selected, teacher=teacher, cartridge=cartridge)


def test_separate_full_and_selected_changes_and_already_correct_cases():
    rows = [row(1, 0, 1, 1, 1, 1), row(2, 0, 1, 0, 0, 1),
            row(3, 0, 1, 0, 1, 0), row(4, 1, 1, 0, 0, 0),
            row(5, 1, 0, 1, 1, 0)]
    result = summarize(rows)
    assert result["full_update"]["new_correct"] == 3
    assert result["full_update"]["new_wrong"] == 1
    assert result["cartridge_update"]["new_correct"] == 1
    assert result["cartridge_update"]["new_wrong"] == 1
    assert result["whole_only"] == {"n": 1, "adapted_full_correct": 1, "cartridge_correct": 0}
    assert result["full_update_new_correct_selected_outcomes"] == dict(
        n=3, selected_already_correct_retained=1, selected_already_correct_lost=0,
        selected_wrong_corrected=1, selected_wrong_still_wrong=1)


def test_rejects_non_binary_scores():
    with pytest.raises(ValueError):
        summarize([row(1, 0, 1, 1, 1, 0.5)])


def test_coverage_is_only_counted_for_the_whole_only_cases():
    rows = [row(1, 1, 1, 0, 0, 0), row(2, 1, 1, 0, 0, 1),
            row(3, 0, 1, 1, 0, 1)]
    for item, value in zip(rows, ("complete", "incomplete", "complete")):
        item["coverage"] = value
    assert summarize(rows)["whole_only_evidence_coverage"] == {"complete": 1, "incomplete": 1}


def test_teacher_format_and_training_changes_are_scored_on_the_same_cases():
    rows = [row(1, 0, 0, 0, 1, 0), row(2, 1, 1, 1, 1, 1)]
    rows[0].update(selective_teacher_format=0, cartridge_teacher_format=1)
    rows[1].update(selective_teacher_format=1, cartridge_teacher_format=1)
    formats = summarize(rows)["teacher_format_controls"]
    assert formats["selected_zero_accuracy"] == 50.
    assert formats["cartridge_accuracy"] == 100.
    assert formats["training_effect"]["new_correct"] == 1
    assert formats["cartridge_format_effect"]["delta_pp"] == 50.
