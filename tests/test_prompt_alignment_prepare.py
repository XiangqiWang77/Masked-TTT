from pathlib import Path
import pytest
from experiments.prompt_alignment import prepare


def protocol():
    development = ["case-" + str(i) for i in range(200)]
    source = dict(development_case_ids=development, holdout_case_ids=["holdout"],
        case_ids=development, models=["qwen", "gpt"], keep_fraction=.1,
        selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25,
        student_rollout_tokens=32, selected_transfer_weight=0,
        reference_answers_used_in_inference=False)
    profile = dict(steps=2, student_rollout_tokens=8, gain_learning_rate=.01,
                   value_learning_rate=.0001, self_question_count=2)
    return source, profile


def test_fixed_protocol_does_not_change_teacher_loss_or_evidence(monkeypatch):
    monkeypatch.setattr(prepare, "validate_partition", lambda *args: None)
    source, profile = protocol()
    result = prepare.validate_protocol(source, profile)
    assert result["case_ids"] == source["case_ids"]
    assert result["align_student_teacher_prompt"] is True
    assert result["selected_transfer_weight"] == 0 and result["online_api_calls"] == 0
    assert result["standalone_answer_format"] == "original" and result["baseline_checkpoint"] == 2
    source["selected_transfer_weight"] = .1
    with pytest.raises(ValueError, match="baseline"):
        prepare.validate_protocol(source, profile)


def test_one_gpu_script_calls_only_aligned_runner():
    root = Path("/tmp/query_cartridges_prompt_aligned_dev_20261104")
    script = prepare.batch_script(root)
    assert "#SBATCH --gpus-per-task=1" in script
    assert "prompt_alignment/execute.py" in script
    assert "HF_HUB_OFFLINE=1" in script
