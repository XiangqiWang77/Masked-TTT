import json

import pytest

from masked_ttt import DecisionConfig, OpenAIChunkSelector, Request
from masked_ttt._layout import selected_read_plan
from masked_ttt.decision_api import MODEL, parse_decisions
from masked_ttt.request import prepare_layout
from masked_ttt.selection import compact_source, project_scores, select_top_fraction


def test_typed_API_selection_has_no_answer_target_and_is_deterministic(tiny_backbone, tmp_path):
    request = Request(
        document="Important evidence. " * 20, question="Which evidence matters?", max_new_tokens=3
    )
    calls = []

    def transport(payload):
        evidence = json.loads(payload["input"])
        assert evidence["original_question"] == request.question
        assert set(evidence) == {
            "original_question",
            "evidence_relevance_policy",
            "background_chunks",
            "candidate_chunks",
        }
        calls.append(payload)
        answers = [
            dict(
                name=q["name"],
                type="choice",
                choice="yes",
                confidence=0.9,
                probabilities=[
                    dict(value="yes", probability=0.9),
                    dict(value="no", probability=0.1),
                ],
            )
            for q in payload["questions"]
        ]
        return dict(model=MODEL, answers=answers, usage=dict(input_tokens=200))

    selector = OpenAIChunkSelector(
        DecisionConfig(cache_dir=str(tmp_path), batch_size=32), transport=transport
    )
    prepared = selector.prepare(tiny_backbone, request)
    assert prepared.request is request and prepared.metadata["complete_original_document"]
    assert prepared.selected == list(
        range(len(prepared.selected))
    )  # Source order breaks equal-score ties.
    assert calls
    again = selector.prepare(tiny_backbone, request)
    assert again.selected == prepared.selected and again.metadata["cache_hits"] > 0


def test_invalid_API_data_is_not_a_keep_all_selection():
    records = parse_decisions(dict(answers=[]), 2)
    with pytest.raises(ValueError, match="Invalid Decisions"):
        select_top_fraction(records, 0.1)


def test_overflow_preserves_verbatim_order_and_maps_scores():
    document = "AAAABBBBCCCC"
    records = [dict(decision_probabilities=dict(yes=p)) for p in [0.8, 0.1, 0.9]]
    text, mapping = compact_source(
        document, [(0, 4), (4, 8), (8, 12)], records, [[0, 1], [2, 3], [4, 5]], 8
    )
    assert text == "AAAA\n\nCCCC"
    mapped = project_scores([(0, 4), (6, 10), (2, 8)], mapping, records)
    assert [r["decision_probabilities"]["yes"] for r in mapped] == pytest.approx([0.8, 0.9, 0.85])


def test_teacher_keeps_original_task_and_question_but_only_selected_source(tiny_backbone):
    request = Request(
        document="Alpha is the first entity. Beta is the second entity. " * 5,
        question="Which entity is first?",
        max_new_tokens=3,
    )
    layout = prepare_layout(tiny_backbone, request)
    plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, [1], [10, 10])
    assert plan.ids[: layout.initial] == tuple(layout.ids[: layout.initial])
    assert plan.ids[plan.document_end :] == tuple(layout.ids[layout.document_end :])
    assert tuple(layout.chunks[1]) == plan.original_document_positions
    assert len(plan.ids) < len(layout.ids)


def test_reference_answer_is_not_an_input_field():
    with pytest.raises(TypeError):
        Request(document="Source", question="Question", answer="gold")
