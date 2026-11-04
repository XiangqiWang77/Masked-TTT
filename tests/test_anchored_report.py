from copy import deepcopy

import pytest

from experiments.anchored_short_verifier.render_report import render


def _result():
    return {
        "metric": "answer_focused_lexical_v2",
        "evaluation_answers_used_only_posthoc": True,
        "mode": "development",
        "expected_per_model": 200,
        "models": {
            "qwen": {
                "rows": [
                    {"task": "qa2", "length": 8192, "whole": 0, "zero": 0,
                     "teacher": 1, "cartridge": 1, "old_cartridge": 0},
                    {"task": "qa1", "length": 16384, "whole": 1, "zero": 1,
                     "teacher": 1, "cartridge": 0, "old_cartridge": None},
                    {"task": "qa1", "length": 16384, "whole": 0, "zero": 0,
                     "teacher": 0, "cartridge": 1, "old_cartridge": 1},
                ]
            }
        },
    }


def test_report_uses_paired_outcomes_and_correct_context_units():
    text = render(_result())
    assert "| qwen | 3 | 33.33 | 33.33 | 66.67 | 66.67 | 1 / 1 | 0.00 | 2 / 1 | 33.33 |" in text
    assert "| qa1 | 2 |" in text and "| qa2 | 1 |" in text
    assert text.index("| qa1 |") < text.index("| qa2 |")
    assert "| 8K | 1 |" in text and "| 16K | 2 |" in text
    assert "Strictly matched original-SD comparison: 2 cases; new right / new wrong = 1 / 0; Δ = 50.00 pp." in text
    assert "Teacher retains the original OLMo response instruction" in text
    assert "A partial run summarizes completed cases only" in text


def test_report_never_treats_missing_or_nonbinary_scores_as_accuracy():
    value = _result()
    value["models"]["qwen"]["rows"][0]["teacher"] = None
    with pytest.raises(ValueError):
        render(value)
    value = _result()
    value["models"]["qwen"]["rows"][0]["old_cartridge"] = .5
    with pytest.raises(ValueError):
        render(value)
    value = deepcopy(_result())
    value["evaluation_answers_used_only_posthoc"] = False
    with pytest.raises(ValueError):
        render(value)


def test_empty_partial_model_keeps_missing_values():
    value = _result()
    value["models"]["qwen"]["rows"] = []
    assert "| qwen | 0 | — | — | — | — | 0 / 0 | — | 0 / 0 | — |" in render(value)
