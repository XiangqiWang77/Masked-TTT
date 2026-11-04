from copy import deepcopy
import pytest
from experiments.value_lr_physical.strong_dev_report import paired, same_baseline


def pair():
    selection = dict(selected_original_chunks=[[1]])
    training = dict(steps=[dict(student_rollout_tokens=[2])],
                    objective="on-policy KL(teacher || student)", selected_transfer_weight=0)
    student = dict(key="case", model="gpt", selection=selection, teacher=dict(tokens=[3]),
        self_study=dict(checks=[]), training=training,
        standalone_cartridge=dict(original_question_controls=dict(zero=dict(tokens=[4]))))
    baseline = dict(key="case", model="gpt", selection=deepcopy(selection), teacher=dict(tokens=[3]),
        self_study=dict(checks=[]), training=deepcopy(training), value_learning_rate=.001,
        gain_learning_rate=.01, standalone_cartridge=dict(paired_prompt_control=dict(
            zero_original=dict(tokens=[4]))))
    return student, baseline


def test_same_start_requires_matched_selected_cache_and_teacher():
    student, baseline = pair()
    same_baseline(student, baseline)
    baseline["teacher"]["tokens"] = [9]
    with pytest.raises(ValueError, match="Baseline"):
        same_baseline(student, baseline)


def test_pair_summary_counts_new_correct_and_wrong():
    result = paired([dict(baseline=0, cartridge=1), dict(baseline=1, cartridge=0),
                     dict(baseline=None, cartridge=1)])
    assert result["n"] == 2 and result["new_correct"] == result["new_wrong"] == 1
    assert result["delta_pp"] == 0
