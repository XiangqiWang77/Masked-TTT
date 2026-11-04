from copy import deepcopy

import pytest

from experiments.value_lr_physical.archive_equivalence import compare


def sample():
    step = dict(teacher_KL_before=.2, gain_gradient_norm=.3, value_gradient_norm=.4,
                gradient_norm=.5, student_rollout_tokens=[1, 2], target_positions=[0, 1],
                question_path='main')
    return dict(architecture='gpt', value_learning_rate=.001, base_steps=2, teacher_tokens=[4], final_tokens=[5],
                full_prediction='blue', full_cache_released=True, weights_unchanged=True,
                cache_unchanged=True, objective='on-policy KL(teacher || student)', steps=[step])


def test_training_trajectory_and_final_tokens_are_compared():
    old=sample();new=deepcopy(old)
    new['steps'][0]['teacher_KL_before']+=1e-8
    assert compare(old,new)['same_final_tokens']
    new['final_tokens']=[6]
    with pytest.raises(ValueError,match='final_tokens'):
        compare(old,new)


def test_nonfinite_or_mismatched_training_signal_is_rejected():
    old=sample();new=deepcopy(old);new['steps'][0]['gain_gradient_norm']=float('nan')
    with pytest.raises(ValueError,match='gain_gradient_norm'):
        compare(old,new)
