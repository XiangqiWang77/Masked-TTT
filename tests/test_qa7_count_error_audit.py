from types import SimpleNamespace

import pytest

from experiments.babilong_linked_selection.qa7_count_error_audit import (
    ARMS, combine_event_status, extra_actor_status, parse_number, summarize,
)


def test_count_parser_handles_final_numbers_and_no_objects():
    focused=lambda text,_task:text
    normalized=lambda text:text.lower().strip()
    assert parse_number("one object",focused,normalized)==1
    assert parse_number("no objects",focused,normalized)==0
    assert parse_number("uncertain",focused,normalized) is None


def test_extra_source_story_coverage_handles_ambiguity_and_excludes_support():
    facts=[(1,"Mary took the apple."),(2,"Mary dropped the apple."),(3,"John left.")]
    document=" ".join(text for _,text in facts)
    def occurrence(text):
        start=document.index(text);return (start,start+len(text))
    helper=SimpleNamespace(ordered_occurrences=lambda _doc,items:[[occurrence(text)] for text in items],
        span_covered=lambda span,ranges:any(a<=span[0] and span[1]<=b for a,b in ranges))
    match=dict(facts=facts,support=[1])
    first=occurrence(facts[0][1]);both=(0,len(document))
    assert extra_actor_status(document,match,[first],"Mary",helper)=="some_missing"
    assert extra_actor_status(document,match,[both],"Mary",helper)=="all_present"
    assert extra_actor_status(document,dict(facts=facts,support=[1,2]),[first],"Mary",helper)=="no_extra_events"
    assert combine_event_status(["all_present","some_missing"])=="ambiguous"
    with pytest.raises(ValueError):combine_event_status([])


def test_count_error_summary_distinguishes_overcount_from_under_and_unparsed():
    def row(source,answer,prediction):
        numbers={arm:prediction for arm in ARMS};scores={arm:int(prediction==answer) for arm in ARMS}
        return dict(source=source,answer=answer,numbers=numbers,scores=scores,
                    coverage="complete",actor_coverage="no_extra_events",
                    study=dict(verified=0,generated_tokens=2,rejections={"invalid_json":1}))
    result=summarize([row("a",0,1),row("a",1,0),row("b",0,0),row("b",1,None)])
    assert result["n"]==4 and result["source_questions"]==2
    assert result["arms"]["teacher"]==dict(correct=1,overcount=1,undercount=1,numeric_match=1,unparsed=1)
    assert result["self_study"]["invalid_json"]==4
    assert result["self_study"]["invalid_json_generated_tokens_median"]==2
