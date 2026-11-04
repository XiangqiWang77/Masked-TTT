from copy import deepcopy
import pytest
from experiments.value_lr_physical.audit_value_strength import RATES, summarize, validate_pair


def candidate(rate):
    control = dict(reference_answers_used=False, teacher_answer_in_student_prompt=False,
        same_selected_kv=True, same_backbone_and_device=True, same_generation_limit=True,
        control_update_steps=0, zero_original=dict(tokens=[1]), zero_teacher_suffix=dict(tokens=[2]))
    return dict(key="case", value_learning_rate=float(rate[3:]), gain_learning_rate=.01,
        training=dict(objective="on-policy KL(teacher || student)", selected_transfer_weight=0,
            value_basis_source="document", teacher_conditioned_on_student_prefix=True,
            student_rollouts_refreshed_each_step=True, steps=[dict(student_rollout_tokens=[3])]),
        selection=dict(reference_answers_used=False, online_api_calls=0, source_case_key="case",
                       complete_original_document=True, selected_original_chunks=[[4]]),
        backbone_unchanged=True, original_document_KV_unchanged=True,
        self_study=dict(reference_answers_used=False, student_answer_hint_used=False, checks=[]),
        teacher=dict(tokens=[4]), standalone_cartridge=dict(paired_prompt_control=control),
        job="123", gpu="GPU", node="node", allocation_restart_count="0")


def test_pair_requires_same_teacher_question_selection_and_allocation():
    results = [candidate(rate) for rate in RATES]
    validate_pair(results, "case")
    for field, value in (("teacher", dict(tokens=[99])), ("node", "other"),
                         ("selection", dict(results[1]["selection"], selected_original_chunks=[[5]]))):
        altered = deepcopy(results)
        altered[1][field] = value
        with pytest.raises(ValueError, match="matched"):
            validate_pair(altered, "case")


def test_rate_summary_counts_both_gains_and_losses():
    rows = []
    for zero, answer in ((0, 1), (1, 0)):
        item = dict(zero=zero)
        for rate in RATES:
            item.update({rate: answer, rate + "_answer_changed": True,
                         rate + "_relative_l2": .01, rate + "_changed": .5})
        rows.append(item)
    summary = summarize(rows)
    assert summary[RATES[0]]["new_correct"] == 1
    assert summary[RATES[0]]["new_wrong"] == 1
    assert summary[RATES[0]]["delta_vs_zero_pp"] == 0
    assert summary[RATES[0]]["changed_answer_sequences"] == 2
