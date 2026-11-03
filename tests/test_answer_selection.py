import pytest

from masked_ttt.answer_selection import select_answer_arm


def test_practice_gate_selects_only_a_trained_verified_practice():
    assert select_answer_arm('practice_gated_format',0,4,True)=='zero_format'
    assert select_answer_arm('practice_gated_format',1,1,True)=='zero_format'
    assert select_answer_arm('practice_gated_format',1,2,True)=='trained_format'
    assert select_answer_arm('practice_gated_format',1,1,False)=='trained_format'
    assert select_answer_arm('practice_gated_format',1,0,False)=='zero_format'
    assert select_answer_arm('original',0,0,False)=='trained'
    assert select_answer_arm('teacher_format',0,0,False)=='trained_format'


@pytest.mark.parametrize('arguments',[
    ('practice_gated_format',True,2,True),('practice_gated_format',1,-1,True),
    ('practice_gated_format',1,2,1),('unknown',0,0,False)])
def test_gate_rejects_invalid_runtime_state(arguments):
    with pytest.raises(ValueError):select_answer_arm(*arguments)
