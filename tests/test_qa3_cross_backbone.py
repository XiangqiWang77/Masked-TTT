import copy

import pytest

from experiments.anchored_short_verifier.audit_qa3_cross_backbone import pair


def result(case, model):
    return {"key": case, "task": "qa3", "model": model, "teacher": {"response_instruction_applied": True},
            "selection": {"complete_original_document": True, "selected_original_chunks": [[2]],
                          "source_case_key": case},
            "training": {"objective": "on-policy KL(teacher || student)", "student_rollout_limit": 8,
                         "steps": [{"question_path": "main", "student_rollout_tokens": list(range(8)),
                                    "target_positions": list(range(8)), "teacher_KL_before": 0.1,
                                    "gain_gradient_norm": 0.2, "value_gradient_norm": 0.3}]}}


def test_requires_matching_full_source_selection_and_decoded_token_pieces():
    case = "babilong-8-qa3-300060"
    qwen = result(case, "Qwen/Qwen3-4B-Instruct-2507")
    olmo = result(case, "allenai/Olmo-3-7B-Instruct")
    decode = lambda tokens: str(tokens[0])
    row = pair(case, qwen, olmo, decode, decode)
    assert row["qwen"]["teacher_KL_before"] == row["olmo"]["teacher_KL_before"] == 0.1
    for changed in ("source", "selection", "prefix", "target"):
        bad = copy.deepcopy(olmo)
        if changed == "source": bad["selection"]["complete_original_document"] = False
        elif changed == "selection": bad["selection"]["selected_original_chunks"] = [[3]]
        elif changed == "prefix": bad["training"]["steps"][0]["student_rollout_tokens"][0] = 9
        else: bad["training"]["steps"][0]["target_positions"] = [0]
        with pytest.raises(ValueError): pair(case, qwen, bad, decode, decode)
