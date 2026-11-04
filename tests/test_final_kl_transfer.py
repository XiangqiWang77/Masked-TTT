import pytest

from experiments.babilong_linked_selection.final_kl_transfer import direction,summarize


def test_direction_uses_same_prefix_difference_and_tolerance():
    assert direction(.5,.4)=='decreased'
    assert direction(.5,.6)=='increased'
    assert direction(.5,.5+1e-7)=='within_tolerance'


def test_summary_counts_kl_and_answer_changes_independently():
    rows=[dict(before=.5,after=.4,zero=0,cartridge=1,teacher=1),
          dict(before=.5,after=.6,zero=1,cartridge=0,teacher=1),
          dict(before=.2,after=.2,zero=1,cartridge=1,teacher=0)]
    result=summarize(rows)
    assert (result['n'],result['kl_decreased'],result['kl_increased'],result['kl_within_tolerance'])==(3,1,1,1)
    assert (result['new_correct'],result['new_wrong'],result['delta_pp'])==(1,1,0.)
    assert result['median_kl_reduction']==pytest.approx(0.)
    assert result['zero_accuracy']==pytest.approx(200/3)
