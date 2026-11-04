import inspect

import pytest

from experiments.babilong_linked_selection.two_hop_selector import two_hop
from experiments.babilong_linked_selection import two_hop_coverage_dev as audit
from masked_ttt.linked_selection import choose


def source(chunks,probabilities):
    document="\n".join(chunks);spans=[];cursor=0
    for text in chunks:
        start=document.index(text,cursor);end=start+len(text);spans.append((start,end));cursor=end
    records=[dict(chunk=i,utility_output_valid=True,decision_probabilities={"yes":p})
             for i,p in enumerate(probabilities)]
    return document,spans,records


def test_two_hop_keeps_the_linking_chunk_and_top_score_anchors():
    chunks=["A neutral record."]*80
    chunks[0]="Daniel gave milk to Mary."
    chunks[10]="Mary passed apple to John."
    chunks[12]="Daniel carried milk to the kitchen."
    chunks[15]="John placed apple in the garden."
    probabilities=[.01]*80
    for i in range(6):probabilities[i]=.95-i*.01
    probabilities[10]=.5;probabilities[12]=.45;probabilities[15]=.005
    document,spans,records=source(chunks,probabilities)
    baseline=choose(document,"Where is the milk?",spans,records,.1,"entity",.25)
    selected=two_hop(document,"Where is the milk?",spans,records,.1)
    assert len(selected)==len(baseline)==8
    assert set(range(6)).issubset(selected)
    assert 10 in baseline and 10 in selected  # The Mary → John bridge remains visible.
    assert 12 in baseline and 12 not in selected and 15 in selected
    assert selected==two_hop(document,"Where is the milk?",spans,records,.1)
    assert two_hop(document,"Where is the milk?",spans,records,.1,require_score_improvement=True)==baseline
    records[15]["decision_probabilities"]["yes"]=.49
    protected=two_hop(document,"Where is the milk?",spans,records,.1,require_score_improvement=True)
    assert 10 in protected and 15 in protected and 12 not in protected


def test_no_room_to_keep_a_bridge_leaves_the_baseline_unchanged():
    chunks=["A neutral record."]*40
    chunks[0]="Daniel gave milk to Mary."
    chunks[10]="Mary passed apple to John."
    chunks[15]="John placed apple in the garden."
    probabilities=[.01]*40
    probabilities[:3]=[.95,.94,.93];probabilities[10]=.5;probabilities[15]=.005
    document,spans,records=source(chunks,probabilities)
    baseline=choose(document,"Where is the milk?",spans,records,.1,"entity",.25)
    assert 10 in baseline
    assert two_hop(document,"Where is the milk?",spans,records,.1)==baseline


def test_empty_or_unsupported_entity_graph_preserves_the_score_selection():
    document,spans,records=source(["A neutral record."]*30,[.01]*30)
    assert two_hop(document,"Where is the milk?",spans,records,.1)==choose(
        document,"Where is the milk?",spans,records,.1,"entity",.25)
    assert "answer" not in inspect.signature(two_hop).parameters
    assert "support" not in inspect.signature(two_hop).parameters
    records[0]["decision_probabilities"]["yes"]=float("nan")
    with pytest.raises(ValueError,match="Decision probability"):
        two_hop(document,"Where is the milk?",spans,records,.1)


def test_coverage_audit_restores_selector_and_uses_posthoc_only(monkeypatch,tmp_path):
    original=audit.base.question_overlap
    stats=dict(net_complete=5,full_only_newly_complete=3,teacher_only_lost_complete=2,
               ambiguous=0,unavailable=0)
    def fake(_workspace):
        assert audit.base.question_overlap is not original
        return dict(cases=200,source_questions=50,fraction=.1,
            groups={"all":dict(entity25={"complete":100},mix=stats,lexical=stats)})
    monkeypatch.setattr(audit.base,"collect",fake)
    data=audit.collect(tmp_path)
    assert audit.base.question_overlap is original
    assert data["groups"]["all"]["two_hop"]==stats and data["decisions"]["two_hop"]["eligible_for_model_test"]
    assert data["selection_accepts_no_labels"] and data["support_and_answers_used_only_posthoc"]
    assert data["online_api_calls"]==0 and data["new_model_calls"]==0
    def fail(_workspace):raise RuntimeError("synthetic failure")
    monkeypatch.setattr(audit.base,"collect",fail)
    with pytest.raises(RuntimeError,match="synthetic failure"):
        audit.collect(tmp_path)
    assert audit.base.question_overlap is original
