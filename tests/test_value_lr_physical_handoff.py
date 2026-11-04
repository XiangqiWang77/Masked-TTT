import json

import pytest

from experiments.value_lr_physical.handoff import submitted_physical_models
from experiments.value_lr_physical.protocol import holdout_manifest
from test_value_lr_physical import fixture_value_development


def test_only_a_matching_submitted_physical_holdout_is_skipped(tmp_path):
    development, report, profile = fixture_value_development()
    manifest = holdout_manifest(development, report, profile, "gpt")
    selection = {"choices": {"gpt": report["models"]["gpt"]["choice"], "qwen": {"value_learning_rate": None}}}
    root = tmp_path / "query_cartridges_value_lr_physical_gpt_holdout_20261104"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert submitted_physical_models(tmp_path, development, selection) == {}
    submission = dict(job_id="12345", gpus=1, model="gpt", cases=800,
                      fixed_value_learning_rate=.001, online_api_calls=0)
    (root / "submission.json").write_text(json.dumps(submission))
    assert submitted_physical_models(tmp_path, development, selection) == {
        "gpt": "physical_holdout_already_submitted"}
    submission["fixed_value_learning_rate"] = .0003
    (root / "submission.json").write_text(json.dumps(submission))
    with pytest.raises(ValueError, match="differs"):
        submitted_physical_models(tmp_path, development, selection)
