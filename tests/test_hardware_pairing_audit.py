from copy import deepcopy

import pytest

from experiments.prompt_alignment.audit_hardware_pairing import pair_flags, summarize


def result(gpu="B200"):
    return dict(key="case", model_short="gpt", gpu=gpu, node="one",
        request_chat_date="2026-11-03",
        selection=dict(selected_original_chunks=[0, 2], selected_chunks=[0, 2],
                       total_chunks=4, selected_tokens=12),
        teacher=dict(tokens=[1, 2]), self_study=dict(checks=[dict(accepted=True)]),
        standalone_cartridge=dict(original_question_controls=dict(zero=dict(tokens=[5, 6]))))


def test_pair_records_each_changed_input_and_compares_gpu_not_node():
    a, b = result(), result("H200")
    b["teacher"]["tokens"] = [1, 3]
    b["self_study"]["checks"][0]["accepted"] = False
    flags = pair_flags(a, b)
    assert not flags["same_gpu"] and flags["same_node"] and flags["same_date"]
    assert flags["teacher_tokens_different"] and flags["practice_different"]
    assert not flags["zero_tokens_different"] and flags["observable_pair_different"]


def test_matching_tokens_do_not_mask_selection_change():
    a = result()
    b = deepcopy(a)
    b["selection"]["selected_chunks"] = [0, 1]
    flags = pair_flags(a, b)
    assert flags["selection_different"] and flags["observable_pair_different"]
    assert summarize([flags])["observable_pair_different"] == 1


def test_missing_gpu_and_mismatched_identity_are_not_silent():
    a, b = result(), result()
    b["gpu"] = ""
    with pytest.raises(ValueError, match="GPU model"):
        pair_flags(a, b)
    b = result()
    b["key"] = "another"
    with pytest.raises(ValueError, match="same case"):
        pair_flags(a, b)
