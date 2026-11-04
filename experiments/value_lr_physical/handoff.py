"""Prevent a legacy launcher from duplicating a physical holdout already submitted."""
import json
from pathlib import Path


def submitted_physical_models(workspace, development, selection):
    workspace = Path(workspace)
    covered = {}
    for model in ("gpt", "qwen"):
        root = workspace / ("query_cartridges_value_lr_physical_" + model + "_holdout_20261104")
        submission_path, manifest_path = root / "submission.json", root / "manifest.json"
        if not submission_path.exists():
            continue
        if not manifest_path.exists():
            raise ValueError("A submitted physical holdout lacks its manifest")
        submitted = json.loads(submission_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        choice = selection["choices"][model]
        if (not str(submitted.get("job_id", "")).isdigit() or submitted.get("gpus") != 1 or
            submitted.get("model") != model or submitted.get("cases") != 800 or
            submitted.get("online_api_calls") != 0 or manifest.get("models") != [model] or
            manifest.get("mode") != "holdout" or manifest.get("case_count") != 800 or
            set(manifest.get("case_ids", [])) != set(development["holdout_case_ids"]) or
            manifest.get("fixed_value_learning_rate") != choice["value_learning_rate"] or
            submitted.get("fixed_value_learning_rate") != choice["value_learning_rate"] or
            manifest.get("selected_transfer_weight") != 0 or manifest.get("online_api_calls") != 0):
            raise ValueError("The existing physical holdout differs from the selected development protocol")
        covered[model] = "physical_holdout_already_submitted"
    return covered
