import pytest

from masked_ttt import Request
from masked_ttt.request import text_sha, prepare_layout
from masked_ttt.frozen_scores import SCHEMA, atomic_json, score_key, load_scores, FrozenJSONSelector


def test_frozen_scores_validate_source_and_never_call_API(tmp_path, tiny_backbone, monkeypatch):
    request = Request(
        document="A project was approved. Another project was cancelled.",
        question="Which was approved?",
        max_new_tokens=4,
    )
    layout = prepare_layout(tiny_backbone, request)
    records = [
        dict(chunk=i, utility_output_valid=True, decision_probabilities=dict(yes=0.8, no=0.2))
        for i in range(len(layout.chunks))
    ]
    path = tmp_path / "scores" / (score_key(request.document, request.question) + ".json")
    atomic_json(
        path,
        dict(
            schema=SCHEMA,
            complete=True,
            document_sha256=text_sha(request.document),
            question_sha256=text_sha(request.question),
            reference_answers_used=False,
            source_spans=layout.spans,
            chunk_sha256=[text_sha(request.document[a:b]) for a, b in layout.spans],
            records=records,
            canonical_tokenizer="test",
            scoring_seconds=1,
            estimated_cost_usd=0.01,
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU selector attempted network access")

    monkeypatch.setattr("masked_ttt.decision_api.DecisionClient.__init__", forbidden)
    prepared = FrozenJSONSelector(tmp_path).prepare(tiny_backbone, request)
    assert prepared.metadata["online_api_calls"] == 0
    assert prepared.layout.ids == layout.ids
    assert prepared.layout.chunks == layout.chunks
    with pytest.raises(FileNotFoundError):
        load_scores(tmp_path, request.document, request.question + " changed")
    with pytest.raises(FileNotFoundError):
        load_scores(tmp_path, request.document + " altered", request.question)


def test_byte_fallback_character_boundary_keeps_original_scores(tmp_path):
    document, question = "abΩcd", "Which symbol?"
    spans = [(0, 3), (2, 5)]
    records = [
        dict(chunk=i, utility_output_valid=True, decision_probabilities=dict(yes=0.8, no=0.2))
        for i in range(2)
    ]
    path = tmp_path / "scores" / (score_key(document, question) + ".json")
    atomic_json(
        path,
        dict(
            schema=SCHEMA,
            complete=True,
            document_sha256=text_sha(document),
            question_sha256=text_sha(question),
            reference_answers_used=False,
            source_spans=spans,
            chunk_sha256=[text_sha(document[a:b]) for a, b in spans],
            records=records,
        ),
    )
    before = path.read_bytes()
    loaded = load_scores(tmp_path, document, question)
    assert loaded["source_spans"] == [list(s) for s in spans]
    assert loaded["records"] == records
    assert path.read_bytes() == before


def test_ascii_chunk_overlap_is_still_rejected(tmp_path):
    document, question = "abXcd", "Which letter?"
    spans = [(0, 3), (2, 5)]
    path = tmp_path / "scores" / (score_key(document, question) + ".json")
    atomic_json(
        path,
        dict(
            schema=SCHEMA,
            complete=True,
            document_sha256=text_sha(document),
            question_sha256=text_sha(question),
            reference_answers_used=False,
            source_spans=spans,
            chunk_sha256=[text_sha(document[a:b]) for a, b in spans],
            records=[
                dict(
                    chunk=i, utility_output_valid=True, decision_probabilities=dict(yes=0.8, no=0.2)
                )
                for i in range(2)
            ],
        ),
    )
    with pytest.raises(ValueError, match="overlapping"):
        load_scores(tmp_path, document, question)
