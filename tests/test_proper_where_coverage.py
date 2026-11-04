from experiments.babilong_linked_selection.audit_proper_where_coverage import summarize


def test_coverage_transition_groups_gains_and_losses():
    rows = [
        dict(old_coverage="incomplete", new_coverage="complete", teacher=1, old_teacher=0,
             zero=1, old_zero=0, cartridge=1, old_cartridge=0),
        dict(old_coverage="complete", new_coverage="complete", teacher=0, old_teacher=1,
             zero=1, old_zero=1, cartridge=1, old_cartridge=1),
    ]
    result = summarize(rows)
    assert result["cases"] == 2
    assert result["coverage_transitions"] == {"complete->complete": 1, "incomplete->complete": 1}
    assert result["teacher_new_correct"] == 1 and result["teacher_new_wrong"] == 1
    assert result["cartridge_gains_by_coverage_transition"] == {"incomplete->complete": 1}
