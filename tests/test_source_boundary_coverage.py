import pytest

from experiments.value_lr_physical.source_boundary_coverage import character_coverage, summarize


def test_selected_source_prefix_lost_to_a_neighboring_native_token():
    document = "aaaBBBccc"
    spans = [(0, 3), (3, 6), (6, 9)]
    chunks = [[0], [1], [2]]
    offsets = [(0, 4), (4, 6), (6, 9)]
    selected = character_coverage(document, spans, chunks, [1], offsets, 0)
    assert selected["requested_nonspace"] == 3
    assert selected["missing_nonspace"] == 1 and selected["missing_selected_spans"] == 1
    assert selected["extra_nonspace"] == 0
    both = character_coverage(document, spans, chunks, [0, 1], offsets, 0)
    assert both["missing_nonspace"] == 0 and both["extra_nonspace"] == 0
    first = character_coverage(document, spans, chunks, [0], offsets, 0)
    assert first["missing_nonspace"] == 0 and first["extra_nonspace"] == 1


def test_counts_document_offsets_and_ignores_whitespace():
    document = "ab CD"
    spans = [(0, 2), (2, 5)]
    chunks = [[0], [1]]
    offsets = [(10, 12), (13, 15)]
    result = character_coverage(document, spans, chunks, [1], offsets, 10)
    assert result["requested_nonspace"] == 2 and result["missing_nonspace"] == 0
    assert result["extra_nonspace"] == 0


def test_disjoint_tokens_are_required_and_summary_counts_cases():
    with pytest.raises(ValueError, match="disjoint"):
        character_coverage("abcd", [(0, 2), (2, 4)], [[0], [0]], [0, 1], [(0, 4)], 0)
    rows = [dict(selected_chunks=1, selected_tokens=1, requested_nonspace=3, missing_nonspace=1,
                 extra_nonspace=0, missing_selected_spans=1),
            dict(selected_chunks=1, selected_tokens=1, requested_nonspace=3, missing_nonspace=0,
                 extra_nonspace=1, missing_selected_spans=0)]
    result = summarize(rows)
    assert result["cases"] == 2 and result["cases_missing_selected_nonspace"] == 1
    assert result["cases_including_adjacent_nonspace"] == 1 and result["total_missing_nonspace"] == 1
    assert result["missing_nonspace_histogram"] == {"0": 1, "1": 1}
