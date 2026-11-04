import json

import pytest

from experiments.gqa_head_gain.report import summarize as original_summary
from experiments.linked_value.detailed_report import collect, render, summarize, validate_rows
from test_study_job import sources


def rows(cases):
    result = []
    for i, case in enumerate(cases):
        parts = case.split("-", 2)
        result.append(dict(case=case, task=parts[2].split("-", 1)[0], length=int(parts[1])*1024,
            whole=int(i % 4 == 0), zero=int(i % 4 == 1), teacher=int(i % 4 == 2),
            cartridge=int(i % 4 in (1, 3)), baseline=None, cartridge_percent=20.))
    return result


def test_counts_gains_and_regressions_by_distinct_source():
    cases = [f"babilong-{length}-qa1-123" for length in (8, 16, 32, 64)]
    records = rows(cases)
    stats = summarize(records)
    assert stats["n"] == 4 and stats["observed_sources"] == 1
    assert stats["versus"]["teacher"]["new_correct"] == 2
    assert stats["versus"]["teacher"]["new_wrong"] == 1
    assert stats["versus"]["teacher"]["source_outcomes"]["mixed"] == 1
    saved = original_summary(records)
    assert validate_rows(records, set(cases), saved) == stats
    saved["accuracy"]["teacher"] = 90
    with pytest.raises(ValueError, match="accuracy"):
        validate_rows(records, set(cases), saved)


def test_partial_and_complete_holdout_tables_exclude_case_ids(tmp_path):
    dev, held = sources()
    development, holdout = rows(dev), rows(held)
    manifest = dict(mode="holdout", models=["gpt"], case_count=800, case_ids=held,
        holdout_case_ids=held, development_case_ids=dev, fixed_value_learning_rate=.001,
        selection_policy="decision_entity_link_v1", selected_transfer_weight=0,
        reference_answers_used_in_inference=False, online_api_calls=0,
        qualitative_case_sources=["qa1-300023", "qa2-300037"])
    base = dict(model="gpt", metric="answer_focused_lexical_v2", online_api_calls=0, errors=[], invalid=[])
    dev_report = dict(base, mode="development", expected=200, rows=development,
                      overall=original_summary(development))
    report = dict(base, mode="holdout", expected=800, rows=holdout[:8], overall=original_summary(holdout[:8]))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "development_snapshot.json").write_text(json.dumps(dev_report))
    (tmp_path / "latest.json").write_text(json.dumps(report))
    partial = collect(tmp_path)
    assert not partial["holdout_complete"] and "combined_1000" not in partial
    assert partial["holdout"]["overall"]["n"] == 8
    assert dev[0] not in json.dumps(partial) and "8K" in render(partial)
    assert "Observed source questions" in render(partial)
    partial["error_count"] = 1
    assert "1 errors" in render(partial)
    report.update(rows=holdout, overall=original_summary(holdout))
    (tmp_path / "latest.json").write_text(json.dumps(report))
    complete = collect(tmp_path)
    assert complete["holdout_complete"] and complete["combined_1000"]["overall"]["n"] == 1000
    assert complete["holdout_excluding_previously_viewed_sources"]["overall"]["n"] == 792
    assert "combined_1000" in render(complete)
    report["rows"][0]["length"] = 999
    (tmp_path / "latest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="task or length"):
        collect(tmp_path)
