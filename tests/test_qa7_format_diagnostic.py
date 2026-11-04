import pytest

from experiments.babilong_linked_selection.qa7_format_diagnostic import format_scores,numeric_answer,summarize


def test_numeric_aliases_and_unknown_text():
    assert numeric_answer('$none$')==0
    assert numeric_answer(' ZERO. ')==0
    assert numeric_answer('one')==1
    assert numeric_answer('two')==2
    assert numeric_answer('one or two') is None


def test_paired_format_scores_and_numeric_changes_are_counted_separately():
    def row(case,raw,formatted):
        return dict(case=case,whole=0,zero=0,teacher=0,cartridge=raw,
                    zero_teacher_format=0,cartridge_teacher_format=formatted)
    rows=[row('a',0,1),row('b',1,0),row('c',0,1)]
    predictions={'a':('one','zero'),'b':('one','none'),'c':('two objects','one')}
    result=summarize(rows,predictions)
    assert result['generic_vs_original']==dict(new_correct=2,new_wrong=1,delta_pp=100/3)
    assert result['score_changes_with_different_number']==2
    assert result['score_changes_not_parsed']==1
    assert result['number_transitions']=={'1 -> 0':2}
    assert format_scores(rows)==dict(n=3,original=100/3,generic=200/3,teacher=0.,
                                   new_correct=2,new_wrong=1,delta_pp=100/3)
    with pytest.raises(ValueError):summarize(rows+[rows[0]],predictions)
