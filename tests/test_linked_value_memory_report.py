import json

import pytest

from experiments.linked_value.memory_report import collect, render
from test_canonical_memory_report import result as memory_result
from test_freeze_linked_value_development import fixture
from test_study_job import sources


def completed_development(root):
    manifest, report = fixture(root)
    _dev, hold = sources()
    manifest.update(development_case_ids=manifest["case_ids"], holdout_case_ids=hold)
    report.update(model="gpt", cached_scores=True, online_api_calls=0)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "latest.json").write_text(json.dumps(report))
    return manifest, report


def test_development_memory_uses_physical_cache_and_no_case_text(tmp_path):
    root = tmp_path / "query_cartridges_linked_value_gpt_dev_20261104"
    root.mkdir()
    manifest, _report = completed_development(root)
    data = collect(root)
    assert data["overall"]["n"] == 200 and data["overall"]["median_cartridge_percent_of_full"] == 10
    assert data["prediction_text_used"] is False and data["reference_answers_used"] is False
    assert manifest["case_ids"][0] not in json.dumps(data)
    assert "90.00%" in render(data)


def test_800_holdout_adds_1000_only_after_all_disjoint_cases(tmp_path):
    dev = tmp_path / "query_cartridges_linked_value_gpt_dev_20261104"
    root = tmp_path / "query_cartridges_linked_value_gpt_holdout_20261104"
    dev.mkdir();root.mkdir()
    manifest, snapshot = completed_development(dev)
    held = manifest["holdout_case_ids"]
    hold_manifest = dict(manifest, mode="holdout", case_count=800, case_ids=held)
    (root / "manifest.json").write_text(json.dumps(hold_manifest))
    (root / "development_snapshot.json").write_text(json.dumps(snapshot))
    rows = []
    for key in held:
        length = int(key.split("-", 2)[1]) * 1024
        rows.append(dict(case=key, task="qa1", length=length))
        saved = memory_result()
        saved.update(key=key, model_short="gpt", task="qa1", length=length,
            training=dict(value_learning_rate=.001, objective="on-policy KL(teacher || student)"),
            selection=dict(selection_policy="decision_entity_link_v1"))
        folder = root / "cases/gpt" / key
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(saved))
    report = dict(mode="holdout", model="gpt", expected=800, metric="answer_focused_lexical_v2",
        fixed_value_learning_rate=.001, cached_scores=True, online_api_calls=0, errors=[], invalid=[], rows=rows[:-1])
    (root / "latest.json").write_text(json.dumps(report))
    assert "combined_1000" not in collect(root)
    report["rows"] = rows
    (root / "latest.json").write_text(json.dumps(report))
    data = collect(root)
    assert data["combined_1000"]["overall"]["n"] == 1000
    assert "combined" in render(data)
    report["invalid"] = [dict(case=held[0])]
    (root / "latest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="holdout case set"):
        collect(root)
