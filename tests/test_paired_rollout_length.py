from copy import deepcopy

import pytest

from experiments.value_lr_physical.paired_rollout_length import pair, summarize


def examples():
    case = "babilong-8-qa3-123"
    def step(tokens, kl, value):
        return dict(question_path="main", student_rollout_tokens=tokens, target_positions=list(range(len(tokens))),
                    teacher_KL_before=kl, gain_gradient_norm=value/2, value_gradient_norm=value)
    def training(limit, tokens, kl, value):
        return dict(objective="on-policy KL(teacher || student)", student_rollout_limit=limit,
                    value_basis_source="document", selected_transfer_weight=0,
                    teacher_conditioned_on_student_prefix=True, student_rollouts_refreshed_each_step=True,
                    steps=[step(tokens, kl, value)])
    common = dict(key=case, task="qa3", backbone_unchanged=True, original_document_KV_unchanged=True,
                  self_study=dict(verified=0, checks=[], reference_answers_used=False),
                  selection=dict(selected_original_chunks=[[1], [3]]), teacher=dict(tokens=[11, 12]))
    old = dict(deepcopy(common), training=training(8, [1, 2], 1e-8, 1e-7))
    old["standalone_cartridge"] = dict(generated_token_ids=[7], prediction="A", paired_prompt_control=dict(
        reference_answers_used=False, teacher_answer_in_student_prompt=False, same_selected_kv=True,
        same_generation_limit=True, zero_original=dict(tokens=[7]), trained_original=dict(tokens=[7], prediction="A")))
    new = dict(deepcopy(common), training=training(32, [1, 2, 3], .4, .8))
    new["standalone_cartridge"] = dict(original_question_controls=dict(reference_answers_used=False,
        teacher_answer_in_student_prompt=False, same_selected_kv=True, same_generation_limit=True,
        zero=dict(tokens=[7])), step_probe=dict(reference_answers_used=False, teacher_answer_in_final_prompt=False,
        same_selected_kv=True, full_kv_used_at_decode=False, checkpoints={"2": dict(total_steps=2, tokens=[8])}))
    score_old = dict(case=case, task="qa3", length=8192, teacher=1, zero=0, cartridge=0)
    score_new = dict(case=case, task="qa3", length=8192, teacher=1, zero=0, step2=1)
    return case, old, new, score_old, score_new


def test_matched_longer_rollout_uses_two_step_predictions_and_gradients():
    row, reasons = pair(*examples())
    assert reasons == [] and row["lengthened"] and row["output_tokens_changed"]
    assert row["first_kl"] == 1e-8 and row["longer_kl"] == .4
    stats = summarize([row])
    assert stats["n"] == 1 and stats["source_questions"] == 1
    assert stats["short_accuracy"] == 0 and stats["long_accuracy"] == 100
    assert stats["new_correct"] == 1 and stats["new_wrong"] == 0


def test_different_evidence_and_student_prefix_are_excluded():
    case, old, new, score_old, score_new = examples()
    new["selection"]["selected_original_chunks"] = [[0]]
    new["training"]["steps"][0]["student_rollout_tokens"] = [4, 5, 6]
    row, reasons = pair(case, old, new, score_old, score_new)
    assert row is None and set(reasons) == {"selection", "student_prefix"}


def test_rejects_wrong_checkpoint_or_score_provenance():
    case, old, new, score_old, score_new = examples()
    new["standalone_cartridge"]["step_probe"]["checkpoints"]["2"]["total_steps"] = 4
    with pytest.raises(ValueError, match="two-step control"):
        pair(case, old, new, score_old, score_new)
    new = examples()[2]
    old["standalone_cartridge"]["generated_token_ids"] = [99]
    with pytest.raises(ValueError, match="two-step control"):
        pair(case, old, new, score_old, score_new)


def test_counts_repeated_lengths_as_one_source_question():
    row, _ = pair(*examples())
    another = dict(row, case="babilong-16-qa3-123", short=1, long=0)
    stats = summarize([row, another])
    assert stats["n"] == 2 and stats["source_questions"] == 1
    assert stats["new_correct"] == 1 and stats["new_wrong"] == 1
