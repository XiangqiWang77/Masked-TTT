import json

import pytest

from experiments.value_lr_physical.canonical_monitor import combine, prepare_holdout
from experiments.value_lr_physical.canonical_report import summarize
from test_canonical_job import sources


def rows(cases, improved=True):
    return [dict(case=case, task="qa1", length=8192, whole=0, original_zero=0, original_teacher=0,
                 original_cartridge=0, canonical_zero=0, canonical_teacher=int(not improved),
                 canonical_cartridge=1, same_teacher_tokens=False, same_zero_tokens=False,
                 same_practice_checks=True, teacher_prompt_length_changed=True) for case in cases]


def report(model_rows, mode):
    return dict(mode=mode, expected_per_model=200 if mode == "development" else 800,
        metric="answer_focused_lexical_v2", errors=[], invalid=[], models={name: dict(rows=data,
        overall=summarize(data)) for name, data in model_rows.items()})


def test_promotes_only_complete_models_that_beat_their_same_run_controls(tmp_path):
    rollout, linked, _profile, _feasibility = sources()
    dev, held = linked["case_ids"], linked["holdout_case_ids"]
    development = tmp_path / "dev"
    (development / "runtime").mkdir(parents=True)
    (development / "runtime" / "frozen.txt").write_text("unchanged")
    manifest = dict(models=["qwen", "gpt"], development_case_ids=dev, holdout_case_ids=held,
                    qualitative_case_sources=["qa1-300023", "qa2-300037"])
    (development / "manifest.json").write_text(json.dumps(manifest))
    data = report({"qwen": rows(dev), "gpt": rows(dev, improved=False)}, "development")
    root, decision = prepare_holdout(tmp_path, development, data)
    assert decision["qwen"]["eligible"] and not decision["gpt"]["eligible"]
    frozen = json.loads((root / "manifest.json").read_text())
    assert frozen["models"] == ["qwen"] and frozen["case_count"] == 800
    assert frozen["case_ids"] == held and (root / "runtime/frozen.txt").read_text() == "unchanged"
    assert "canonical_execute.py" in (root / "job.sbatch").read_text()
    data["models"]["qwen"]["overall"]["accuracy"]["whole"] = 100
    with pytest.raises(ValueError, match="scores"):
        prepare_holdout(tmp_path, development, data)


def test_combined_report_checks_1000_cases_and_excludes_previously_viewed_sources():
    _rollout, linked, _profile, _feasibility = sources()
    dev, held = linked["case_ids"], linked["holdout_case_ids"]
    manifest = dict(models=["qwen"], development_case_ids=dev, holdout_case_ids=held,
                    qualitative_case_sources=["qa1-300023", "qa2-300037"])
    result = combine(manifest, report({"qwen": rows(dev)}, "development"),
                     report({"qwen": rows(held)}, "holdout"))
    item = result["models"]["qwen"]
    assert item["overall"]["n"] == 1000
    assert item["holdout_excluding_previously_viewed_sources"]["n"] == 792
    bad = rows(held)
    bad[0]["case"] = dev[0]
    with pytest.raises(ValueError):
        combine(manifest, report({"qwen": rows(dev)}, "development"),
                report({"qwen": bad}, "holdout"))


def test_another_model_failure_cannot_block_a_valid_promoted_model():
    _rollout, linked, _profile, _feasibility = sources()
    dev, held = linked["case_ids"], linked["holdout_case_ids"]
    manifest = dict(models=["qwen"], development_case_ids=dev, holdout_case_ids=held,
                    qualitative_case_sources=["qa1-300023", "qa2-300037"])
    development = report({"qwen": rows(dev)}, "development")
    development["errors"] = [dict(model="gpt")]
    assert combine(manifest, development, report({"qwen": rows(held)}, "holdout"))["models"]["qwen"]["overall"]["n"] == 1000
    development["errors"] = [dict(model="qwen")]
    with pytest.raises(ValueError, match="valid splits"):
        combine(manifest, development, report({"qwen": rows(held)}, "holdout"))
