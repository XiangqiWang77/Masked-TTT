import pytest

from experiments.babilong_linked_selection.focused_question_outcomes import summarize


def test_new_support_is_partitioned_by_saved_teacher_and_full_outcomes():
    rows = [dict(coverage={"proper_where": "incomplete", "lexical": "complete"}, selection_changed={"proper_where": False},
                 source="first", teacher=0, whole=1),
            dict(coverage={"proper_where": "incomplete", "lexical": "complete"}, selection_changed={"proper_where": False},
                 source="second", teacher=1, whole=0),
            dict(coverage={"proper_where": "complete", "lexical": "incomplete"}, selection_changed={"proper_where": False},
                 source="third", teacher=1, whole=1)]
    result = summarize(rows)
    assert result["newly_complete"] == 2 and result["lost_complete"] == 1
    assert result["newly_complete_old_teacher_wrong"] == 1
    assert result["newly_complete_old_full_correct_teacher_wrong"] == 1
    assert result["newly_complete_old_teacher_correct"] == 1


def test_does_not_assume_saved_answers_match_an_old_selection():
    row = dict(coverage={"proper_where": "incomplete", "lexical": "complete"},
               selection_changed={"proper_where": True}, source="first", teacher=0, whole=1)
    with pytest.raises(ValueError, match="Saved baseline answers"):
        summarize([row])
