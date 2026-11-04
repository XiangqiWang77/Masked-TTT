import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "experiments/value_lr_physical/audit_prompt_format.py"
spec = importlib.util.spec_from_file_location("prompt_format_audit", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    control = dict(same_selected_kv=True, same_backbone_and_device=True,
                   same_generation_limit=True, control_update_steps=0,
                   teacher_answer_in_student_prompt=False, reference_answers_used=False,
                   zero_original=dict(tokens=[1, 2]), zero_teacher_suffix=dict(tokens=[3, 2]),
                   trained_original=dict(tokens=[3, 2]))
    result = dict(standalone_cartridge=dict(paired_prompt_control=control),
                  training=dict(objective="on-policy KL(teacher || student)",
                                steps=[dict(student_rollout_tokens=[1])]),
                  teacher=dict(tokens=[3, 2]))
    return result, dict(zero=0, trained=1, teacher=1, zero_format=1)


def test_paired_gains_and_format_bins():
    result, scores = fixture()
    row = audit.read_row(result, scores)
    stats = audit.summarize([row])
    assert stats["overall"]["sd_new_correct"] == 1
    assert stats["overall"]["format_new_correct"] == 1
    assert stats["overall"]["sd_delta_pp"] == 100
    assert stats["by_original_format_first_token_difference"]["different"]["n"] == 1
    assert stats["by_original_format_first_token_difference"]["same"]["n"] == 0


def test_reject_unmatched_or_missing_inputs():
    result, scores = fixture()
    result["standalone_cartridge"]["paired_prompt_control"]["same_selected_kv"] = False
    with pytest.raises(ValueError, match="paired output"):
        audit.read_row(result, scores)
    result, scores = fixture()
    result["teacher"]["tokens"] = []
    with pytest.raises(ValueError, match="token sequences"):
        audit.read_row(result, scores)


def test_reject_non_binary():
    result, scores = fixture()
    scores["trained"] = 2
    with pytest.raises(ValueError, match="binary"):
        audit.read_row(result, scores)
