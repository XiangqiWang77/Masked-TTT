import pytest

from experiments.babilong_linked_selection.paired_attribution import summarize, validate_pair


def fixture_pair():
    def result(chunks, trained):
        control = dict(same_selected_kv=True, same_backbone_and_device=True, same_generation_limit=True,
            control_update_steps=0, teacher_answer_in_student_prompt=False, reference_answers_used=False,
            zero_original=dict(prediction="wrong"), trained_original=dict(prediction=trained),
            zero_teacher_suffix=dict(prediction="right"), trained_teacher_suffix=dict(prediction="right"))
        return dict(key="case", model="qwen", task="qa1",
            selection=dict(complete_original_document=True, reference_answers_used=False, online_api_calls=0,
                           source_case_key="case", selected_original_chunks=[[i] for i in chunks], total_chunks=20),
            training=dict(selected_transfer_weight=0, objective="on-policy KL(teacher || student)",
                          student_rollout_limit=8, optimizer="Adam"),
            self_study=dict(reference_answers_used=False, student_answer_hint_used=False),
            teacher_answer_in_final_prompt=False, backbone_unchanged=True, original_document_KV_unchanged=True,
            teacher=dict(prediction="right"), self_judge=dict(native_prediction="same"),
            standalone_cartridge=dict(paired_prompt_control=control))
    old, new = result([1, 2], "wrong"), result([1, 3], "right")
    coverage = dict(score10=dict(status="incomplete", selected=[1, 2]),
                    entity10=dict(status="complete", selected=[1, 3]))
    row = dict(case="case", task="qa1", length=8192, baseline_source="full_1000", same_full_answer=True, whole=0,
        baseline10=0, zero=0, teacher=1, cartridge=1, zero_teacher_format=1, cartridge_teacher_format=1)
    reference = dict(task="qa1", answers=["right"])
    grade = lambda text, answers, benchmark, task: (int(text in answers), True)
    return row, coverage, old, new, reference, grade


def test_pair_compares_only_matching_sources_and_common_judge():
    row, coverage, old, new, reference, grade = fixture_pair()
    result = validate_pair("case", row, coverage, old, new, reference, grade)
    assert result["coverage"] == "newly_complete"
    assert result["baseline10"] == 0 and result["cartridge"] == 1
    assert "prediction" not in result and "answers" not in result
    summary = summarize([result])
    assert summary["pairs"]["selection_and_method"]["delta_pp"] == 100
    assert summary["pairs"]["sd_original_format"]["new_correct"] == 1
    assert summary["pairs"]["sd_teacher_format"]["delta_pp"] == 0
    assert summary["complementarity"]["teacher_only"] == dict(
        n=1, cartridge_correct=1, cartridge_teacher_format_correct=1,
        new_correct_vs_zero=1, new_wrong_vs_zero=0)
    assert summarize([])["pairs"]["selection_and_method"]["delta_pp"] is None


@pytest.mark.parametrize("change", ["wrong_chunk", "teacher_leak", "practice_leak", "model_changed",
                                   "different_task", "wrong_score", "different_full"])
def test_pair_rejects_unmatched_or_invalid_controls(change):
    row, coverage, old, new, reference, grade = fixture_pair()
    if change == "wrong_chunk":
        coverage["entity10"]["selected"] = [1, 4]
    elif change == "teacher_leak":
        new["standalone_cartridge"]["paired_prompt_control"]["teacher_answer_in_student_prompt"] = True
    elif change == "practice_leak":
        new["self_study"]["reference_answers_used"] = True
    elif change == "model_changed":
        new["backbone_unchanged"] = False
    elif change == "different_task":
        new["task"] = "qa2"
    elif change == "wrong_score":
        row["cartridge"] = 0
    else:
        new["self_judge"]["native_prediction"] = "different"
    with pytest.raises(ValueError):
        validate_pair("case", row, coverage, old, new, reference, grade)
