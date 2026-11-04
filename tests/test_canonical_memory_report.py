import json

import pytest

from experiments.value_lr_physical.canonical_memory_report import collect, memory_record, render, summarize


def result(full=1000, cartridge=100):
    return dict(memory=dict(cache_sizes_exclude_question_and_generated_suffix=True,
        gpu_peak_includes_model_and_training=True,
        full_cache_release_before_standalone=dict(full_cache_tensors_live_before_standalone=0),
        full_document_kv_bytes=full, model_resident_gpu_bytes=10000, paired_run_peak_gpu_allocated_bytes=20000),
        standalone_cartridge=dict(memory=dict(kv_only_materialized=True, cartridge_adapter_bytes=0,
        raw_full_document_kv_used_at_decode=False, selected_kv_representation="independently_prefilled_selected_text",
        additional_source_kv_bytes_before_decode=0, additional_training_adapter_bytes_before_decode=0,
        standalone_cartridge_bytes=cartridge, source_selected_kv_bytes=80,
        live_document_state_before_decode_bytes=cartridge, cuda_allocated_before_decode_bytes=11000)))


def test_requires_physical_release_and_measures_document_kv_only():
    saved = result()
    row = memory_record(saved)
    assert row == dict(full=1000, cartridge=100, selected_source=80,
                       model_gpu=10000, training_peak_gpu=20000, decode_gpu=11000)
    saved["standalone_cartridge"]["memory"]["additional_source_kv_bytes_before_decode"] = 1
    with pytest.raises(ValueError, match="released"):
        memory_record(saved)
    saved = result()
    saved["standalone_cartridge"]["memory"]["live_document_state_before_decode_bytes"] += 1
    with pytest.raises(ValueError, match="outside the cartridge"):
        memory_record(saved)


def test_ratio_is_median_of_per_case_ratios_not_ratio_of_medians():
    rows = [memory_record(result(100, 50)), memory_record(result(1000, 500)),
            memory_record(result(10000, 100))]
    data = summarize(rows)
    assert data["median_full_document_kv_bytes"] == 1000
    assert data["median_standalone_cartridge_bytes"] == 100
    assert data["median_cartridge_percent_of_full"] == 50
    assert data["median_document_kv_reduction_percent"] == 50
    assert summarize([]) == {"n": 0}
    rows[0]["model_gpu"] = None
    with pytest.raises(ValueError, match="Partial CUDA"):
        summarize(rows)


def test_collects_only_scored_cases_and_exports_aggregates(tmp_path):
    key = "babilong-8-qa1-300001"
    manifest = dict(mode="development", case_count=200, case_ids=[key], models=["qwen"])
    report = dict(metric="answer_focused_lexical_v2", mode="development", expected_per_model=200,
        cached_scores=True, online_api_calls=0,
        models={"qwen": dict(rows=[dict(case=key, task="qa1", length=8192)])})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "latest.json").write_text(json.dumps(report))
    folder = tmp_path / "cases/qwen" / key
    folder.mkdir(parents=True)
    saved = result()
    saved.update(key=key, model_short="qwen", task="qa1", length=8192,
                 prediction="not included in aggregate")
    (folder / "result.json").write_text(json.dumps(saved))
    data = collect(tmp_path)
    assert data["models"]["qwen"]["overall"]["n"] == 1
    assert data["reference_answers_used"] is False and data["prediction_text_used"] is False
    assert key not in json.dumps(data) and "not included in aggregate" not in json.dumps(data)
    assert "90.00%" in render(data)
    report["models"]["qwen"]["rows"].append(report["models"]["qwen"]["rows"][0])
    (tmp_path / "latest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="duplicated"):
        collect(tmp_path)


def test_complete_disjoint_holdout_adds_1000_case_memory_summary(tmp_path):
    from test_study_job import sources
    development, holdout = sources()
    dev_root = tmp_path / "query_cartridges_canonical_pair_dev_20261104"
    root = tmp_path / "query_cartridges_canonical_pair_holdout_20261104"
    root.mkdir()
    manifest = dict(mode="holdout", case_count=800, case_ids=holdout, models=["qwen"],
                    development_case_ids=development, holdout_case_ids=holdout)
    (root / "manifest.json").write_text(json.dumps(manifest))
    def build_rows(base, cases):
        rows = []
        for case in cases:
            length = int(case.split("-", 2)[1]) * 1024
            row = dict(case=case, task="qa1", length=length)
            rows.append(row)
            folder = base / "cases/qwen" / case
            folder.mkdir(parents=True)
            saved = result()
            saved.update(key=case, model_short="qwen", task="qa1", length=length)
            (folder / "result.json").write_text(json.dumps(saved))
        return rows
    dev_rows, held_rows = build_rows(dev_root, development), build_rows(root, holdout)
    base = dict(metric="answer_focused_lexical_v2", cached_scores=True, online_api_calls=0, errors=[], invalid=[])
    snapshot = dict(base, mode="development", expected_per_model=200, models={"qwen": dict(rows=dev_rows)})
    (root / "development_snapshot.json").write_text(json.dumps(snapshot))
    report = dict(base, mode="holdout", expected_per_model=800, models={"qwen": dict(rows=held_rows[:-1])})
    (root / "latest.json").write_text(json.dumps(report))
    assert "combined_1000" not in collect(root)["models"]["qwen"]
    report["models"]["qwen"]["rows"] = held_rows
    (root / "latest.json").write_text(json.dumps(report))
    complete = collect(root)
    assert complete["models"]["qwen"]["combined_1000"]["overall"]["n"] == 1000
    assert "qwen combined" in render(complete)
    report["errors"] = [dict(model="qwen")]
    (root / "latest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="holdout case set"):
        collect(root)
