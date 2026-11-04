from pathlib import Path
import pytest
from experiments.value_lr_physical import strong_dev_prepare


def test_manifest_keeps_original_sd_and_fixed_model_rate(monkeypatch):
    monkeypatch.setattr(strong_dev_prepare, "validate_partition", lambda *_: None)
    ids = ["case" + str(i) for i in range(200)]
    source = dict(case_ids=ids, holdout_case_ids=["holdout"], keep_fraction=.1,
        selection_policy="cached_decision_score10", student_rollout_tokens=32,
        selected_transfer_weight=0., reference_answers_used_in_inference=False, online_api_calls=0)
    profile = dict(steps=2, student_rollout_tokens=8, value_learning_rate=.0001,
                   gain_learning_rate=.01, self_question_count=2)
    result = strong_dev_prepare.make_manifest(source, profile)
    assert result["models"] == ["gpt"] and result["case_ids"] == ids
    assert result["fixed_value_learning_rate"] == .003
    assert result["student_rollout_tokens"] == 8 and result["selected_transfer_weight"] == 0
    assert result["holdout_evaluated_at_selection"] is False and result["online_api_calls"] == 0
    source["selected_transfer_weight"] = .1
    with pytest.raises(ValueError, match="protocol"):
        strong_dev_prepare.make_manifest(source, profile)


def test_development_job_is_one_gpu_with_short_backfill():
    script = strong_dev_prepare.batch_script(Path("/tmp/query_cartridges_strong_value_gpt_dev_20261104"))
    assert "#SBATCH --gpus-per-task=1" in script
    assert "#SBATCH --time-min=00:20:00" in script
    assert "value_lr_physical/strong_dev_execute.py" in script
    assert "HF_HUB_OFFLINE=1" in script
