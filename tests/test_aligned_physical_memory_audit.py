import json

import pytest

from experiments.prompt_alignment.physical_memory_audit import collect
from test_canonical_memory_report import result as memory_result
from test_study_job import sources


def make_case(root):
    cases, _ = sources()
    manifest = dict(mode="development", case_count=200, case_ids=cases, models=["qwen", "gpt"],
                    reference_answers_used_in_inference=False, online_api_calls=0)
    report = dict(metric="answer_focused_lexical_v2", expected_per_model=200, errors=[], invalid=[], models={})
    for model in manifest["models"]:
        rows = []
        for case in cases:
            row = dict(case=case, task="qa1", length=8192)
            rows.append(row)
            saved = memory_result(1000, 100 if model == "qwen" else 200)
            saved.update(key=case, model_short=model, task="qa1", length=8192,
                training=dict(objective="on-policy KL(teacher || student)",
                    student_question_matches_teacher=True, selected_transfer_weight=0),
                selection=dict(selection_policy="decision_entity_link_v1", reference_answers_used=False,
                               online_api_calls=0))
            folder = root / "cases" / model / case
            folder.mkdir(parents=True)
            (folder / "result.json").write_text(json.dumps(saved))
        report["models"][model] = dict(candidate_completed=200, candidate_rows=rows)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "latest.json").write_text(json.dumps(report))
    (root / "finished.json").write_text(json.dumps(dict(all_complete=True, completed={"qwen": 200, "gpt": 200})))
    return manifest, report


def test_collects_completed_released_memory_without_case_text(tmp_path):
    manifest, _report = make_case(tmp_path)
    result = collect(tmp_path)
    assert result["models"]["qwen"]["overall"]["n"] == 200
    assert result["models"]["qwen"]["overall"]["median_cartridge_percent_of_full"] == 10
    assert result["models"]["gpt"]["overall"]["median_cartridge_percent_of_full"] == 20
    assert result["reference_answers_used"] is False and result["model_calls"] == 0
    assert manifest["case_ids"][0] not in json.dumps(result)


def test_rejects_missing_cases_and_unreleased_memory(tmp_path):
    manifest, report = make_case(tmp_path)
    report["models"]["qwen"]["candidate_rows"].pop()
    (tmp_path / "latest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact case set"):
        collect(tmp_path)
    report["models"]["qwen"]["candidate_rows"].append(
        dict(case=manifest["case_ids"][-1], task="qa1", length=8192))
    (tmp_path / "latest.json").write_text(json.dumps(report))
    path = tmp_path / "cases/qwen" / manifest["case_ids"][0] / "result.json"
    saved = json.loads(path.read_text())
    saved["standalone_cartridge"]["memory"]["additional_training_adapter_bytes_before_decode"] = 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="released"):
        collect(tmp_path)
