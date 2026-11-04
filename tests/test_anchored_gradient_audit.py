import json

import pytest

from experiments.anchored_short_verifier.audit_pair_gradients import audit, compare_initial_step, render, _summary


def _result(loss=1.0, gain=2.0, value=3.0, tokens=(1, 2)):
    return {"training": {"steps": [{
        "question_path": "main",
        "student_rollout_tokens": list(tokens),
        "target_positions": [0, 1],
        "teacher_KL_before": loss,
        "gain_gradient_norm": gain,
        "value_gradient_norm": value,
    }]}}


def test_first_update_comparison_uses_same_input_and_finite_signals():
    result = compare_initial_step(_result(), _result(loss=1.1, value=3.3, tokens=(3, 4)))
    assert not result["same_student_rollout"] and result["same_target_positions"]
    assert result["gain_gradient_relative_difference"] == 0
    assert result["kl_relative_difference"] == pytest.approx(.1 / 1.1)
    assert result["value_gradient_relative_difference"] == pytest.approx(.3 / 3.3)
    unmatched = _summary([result])
    assert unmatched["n"] == 1 and unmatched["same_first_rollout_and_positions"] == 0
    assert unmatched["median_kl_relative_difference"] is None
    with pytest.raises(ValueError):
        compare_initial_step(_result(loss=float("nan")), _result())
    other = _result()
    other["training"]["steps"][0]["question_path"] = "practice"
    with pytest.raises(ValueError):
        compare_initial_step(other, _result())


def test_audit_counts_only_available_matched_baselines(tmp_path):
    candidate, qwen, gpt, lr = (tmp_path / name for name in ("new", "qwen", "gpt", "lr"))
    for path, result in [
        (candidate / "cases/qwen/ok/result.json", _result(loss=1.1)),
        (qwen / "cases/ok/weight_0.00.json", _result()),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    scored = {"models": {"qwen": {"rows": [
        {"case": "ok", "baseline_source": "full_1000"},
        {"case": "missing", "baseline_source": "full_1000"},
        {"case": "unmatched", "baseline_source": None},
    ]}}}
    output = audit(scored, candidate, {"qwen": qwen, "gpt": gpt}, lr)
    summary = output["models"]["qwen"]["summary"]
    assert summary["n"] == 1 and summary["same_first_rollout_and_positions"] == 1
    assert summary["cases_kl_relative_difference_over_1_percent"] == 1
    assert len(output["unavailable_cases"]) == 1
    assert not output["reference_answers_used"] and output["online_api_calls"] == 0
    assert "| qwen | 1 | 1 | 9.091% | 0.000% | 0.000% |" in render(output)
