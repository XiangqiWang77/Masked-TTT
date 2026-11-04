import pytest

from experiments.value_lr_physical.freeze_basis_ablation import audit_pair, summarize


def paired_fixture():
    def arm(mode, prediction, energy):
        pair = dict(teacher_answer_in_student_prompt=False, reference_answers_used=False,
            same_selected_kv=True, same_generation_limit=True, same_backbone_and_device=True, control_update_steps=0,
            zero_original=dict(tokens=[1], prediction="wrong"),
            zero_teacher_suffix=dict(tokens=[2]), trained_original=dict(prediction=prediction))
        return dict(key="case", basis_source=mode, model="qwen", task="qa1", length=8192,
            job="job", gpu="gpu", node="node",
            training=dict(value_basis_source=mode, objective="on-policy KL(teacher || student)",
                          selected_transfer_weight=0, steps=[{}]),
            selection=dict(source_case_key="case", reference_answers_used=False, online_api_calls=0,
                           complete_original_document=True, selected_original_chunks=[[1], [2]]),
            basis_probe=dict(rank=8, selected_tokens=10, sampled_tokens=10, mean_selected_energy_full_basis=energy,
                             model_or_cache_modified=False, reference_answers_used=False),
            memory=dict(full_cache_release_before_standalone=dict(requested=True,
                                                                 full_cache_tensors_live_before_standalone=0)),
            backbone_unchanged=True, original_document_KV_unchanged=True,
            self_study=dict(reference_answers_used=False, student_answer_hint_used=False),
            teacher_answer_in_final_prompt=False, teacher=dict(tokens=[3], prediction="right"),
            self_judge=dict(native_prediction="right"), standalone_cartridge=dict(paired_prompt_control=pair))
    document = arm("document", "right", .3)
    selected = arm("selected", "wrong", .4)
    row = dict(task="qa1", length=8192, whole=1, zero=0, teacher=1, document=1, selected=0,
               document_basis_energy=.3, selected_basis_energy=.4)
    reference = dict(task="qa1", answers=["right"])
    grade = lambda text, answers, benchmark, task: (int(text in answers), True)
    return row, document, selected, reference, grade


def test_basis_snapshot_audits_controls_and_keeps_no_answers():
    row, document, selected, reference, grade = paired_fixture()
    result = audit_pair("case", row, document, selected, reference, grade)
    assert "prediction" not in result and "answers" not in result
    summary = summarize([result])
    assert summary["selected_vs_document"] == dict(new_correct=0, new_wrong=1, delta_pp=-100)
    assert summary["median_document_basis_energy"] == .3
    assert summary["median_selected_basis_energy"] == .4


@pytest.mark.parametrize("change", ["selection", "teacher", "label_leak", "cache", "device", "energy", "score"])
def test_basis_snapshot_rejects_changed_inputs_or_controls(change):
    row, document, selected, reference, grade = paired_fixture()
    if change == "selection":
        selected["selection"]["selected_original_chunks"] = [[1], [3]]
    elif change == "teacher":
        selected["teacher"]["tokens"] = [99]
    elif change == "label_leak":
        selected["self_study"]["reference_answers_used"] = True
    elif change == "cache":
        selected["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] = 1
    elif change == "device":
        selected["node"] = "different"
    elif change == "energy":
        selected["basis_probe"]["mean_selected_energy_full_basis"] = 1.1
    else:
        row["document"] = 0
    with pytest.raises(ValueError):
        audit_pair("case", row, document, selected, reference, grade)
