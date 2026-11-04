import json

import pytest

from experiments.linked_value import source_stability
from experiments.linked_value.source_stability import source_pairs, summarize
from test_study_job import sources


def row(source, teacher, cartridge, *, baseline=0, zero=0, whole=0, coverage="complete", verified=0):
    return dict(source=source, teacher=teacher, cartridge=cartridge, baseline=baseline,
                zero=zero, whole=whole, coverage=coverage, verified=verified)


def test_counts_mixed_gains_and_regressions_by_source_without_double_counting():
    rows = [row("a", 0, 1), row("a", 1, 0), row("b", 0, 1),
            row("c", 1, 0), row("d", 0, 0)]
    result = source_pairs(rows, "teacher")
    assert result["n"] == 5 and result["source_questions"] == 4
    assert result["new_correct"] == 2 and result["new_wrong"] == 2 and result["delta_pp"] == 0
    assert result["source_outcomes"] == dict(improved_only=1, regressed_only=1, mixed=1, unchanged=1)
    assert result["changed_lengths_per_source"] == {"0": 1, "1": 2, "2": 1, "3": 0, "4": 0}


def test_unmatched_baselines_do_not_enter_paired_denominator():
    rows = [row("a", 0, 1, baseline=None), row("b", 0, 1, baseline=0), row("c", 0, 0, baseline=1)]
    result = source_pairs(rows, "baseline")
    assert result["n"] == 2 and result["new_correct"] == 1 and result["new_wrong"] == 1
    rows[0]["coverage"] = "other"
    with pytest.raises(ValueError, match="coverage"):
        summarize(rows)


def test_joins_only_same_source_chunks_and_exports_aggregates(tmp_path, monkeypatch):
    keys, _ = sources()
    rows, annotations = [], []
    for key in keys:
        length = int(key.split("-", 2)[1]) * 1024
        rows.append(dict(case=key, task="qa1", length=length, whole=0, teacher=0,
                         zero=0, cartridge=1, baseline=0))
        annotations.append(dict(case=key, task="qa1", length=length,
                                outcomes=dict(entity10=dict(status="complete", selected=[0, 2]))))
        folder = tmp_path / "cases/gpt" / key
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(dict(selection=dict(selected_original_chunks=[[0], [2]]),
                                                            self_study=dict(verified=0))))
    (tmp_path / "latest.json").write_text(json.dumps(dict(rows=rows)))
    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps(dict(rows=annotations, answers_and_annotations_used_only_posthoc=True,
                                        online_api_calls=0)))
    monkeypatch.setattr(source_stability, "validate_development", lambda _: dict(metric="answer_focused_lexical_v2"))
    data = source_stability.collect(tmp_path, coverage)
    assert data["cases"] == 200 and data["overall"]["source_questions"] == 50
    assert data["overall"]["versus"]["teacher"]["new_correct"] == 200
    assert keys[0] not in json.dumps(data) and data["new_model_calls"] == 0
    annotations[0]["outcomes"]["entity10"]["selected"] = [1]
    coverage.write_text(json.dumps(dict(rows=annotations, answers_and_annotations_used_only_posthoc=True,
                                        online_api_calls=0)))
    with pytest.raises(ValueError, match="selection differ"):
        source_stability.collect(tmp_path, coverage)
