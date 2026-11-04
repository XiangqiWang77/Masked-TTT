from copy import deepcopy

import pytest

from experiments.babilong_linked_selection.question_overlap_selector import question_overlap
from masked_ttt.linked_selection import choose


def example():
    chunks = ["ordinary passage about unrelated events"]*9 + ["Alice travelled home. Alice slept."]
    document = "\n".join(chunks)
    spans, cursor = [], 0
    for chunk in chunks:
        spans.append((cursor, cursor+len(chunk)))
        cursor += len(chunk)+1
    records = [dict(chunk=i, utility_output_valid=True,
                    decision_probabilities=dict(yes=1-i*.05)) for i in range(10)]
    return document, spans, records


@pytest.mark.parametrize("mode", ["lexical", "mix", "proper_where"])
def test_question_overlap_preserves_budget_and_score_anchors_without_labels(mode):
    document, spans, records = example()
    original = deepcopy(records)
    baseline = choose(document, "Where is Alice?", spans, records, .4, "entity", .25)
    selected = question_overlap(document, "Where is Alice?", spans, records, .4, mode=mode)
    assert len(selected) == len(baseline) == 4
    assert {0, 1, 2}.issubset(selected)
    assert 9 in selected and 9 not in baseline
    assert records == original


def test_invalid_or_too_small_selection_uses_safe_existing_policy():
    document, spans, records = example()
    assert question_overlap(document, "Where is Alice?", spans, records, .1) == [0]
    with pytest.raises(ValueError, match="Unsupported"):
        question_overlap(document, "Where is Alice?", spans, records, .4, mode="other")


def test_person_location_routing_depends_only_on_question_text():
    document, spans, records = example()
    for question in ("Where is the milk?", "Where was Alice before the kitchen?", "How many objects is Alice carrying?"):
        assert question_overlap(document, question, spans, records, .4, mode="proper_where") == \
               choose(document, question, spans, records, .4, "entity", .25)
