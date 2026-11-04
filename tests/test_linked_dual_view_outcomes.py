import pytest

from experiments.babilong_linked_selection.dual_view_outcomes import summarize


def test_paired_complementarity_uses_each_method_on_the_same_case():
    rows = [dict(case="babilong-8-qa3-1", whole=1, zero=0, teacher=0, cartridge=1),
            dict(case="babilong-16-qa3-1", whole=0, zero=1, teacher=1, cartridge=0),
            dict(case="babilong-32-qa3-2", whole=0, zero=1, teacher=0, cartridge=1)]
    stats = summarize(rows)
    assert stats["n"] == 3 and stats["source_questions"] == 2
    assert stats["whole_only"] == dict(n=1, cartridge_correct=1)
    assert stats["selected_only"] == dict(n=2, cartridge_correct=1)
    assert stats["versus_zero"]["new_correct"] == 1 and stats["versus_zero"]["new_wrong"] == 1
    rows[0]["whole"] = 2
    with pytest.raises(ValueError):
        summarize(rows)
