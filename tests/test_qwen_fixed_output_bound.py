from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.babilong_linked_selection.qwen_fixed_output_bound import analyze


SNAPSHOT = Path(__file__).resolve().parents[1] / "docs/analysis/qwen-linked-1000-20261104/summary.json"


def test_fixed_outputs_have_the_expected_answer_aware_upper_bound():
    result = analyze(json.loads(SNAPSHOT.read_text()))
    raw, formatted = result["summaries"]["original_question"], result["summaries"]["teacher_requirements"]
    assert raw["oracle_correct"] == 608
    assert formatted["oracle_correct"] == 621
    assert formatted["gap_to_teacher_plus_three_pp"] == pytest.approx(2.2)
    assert result["summaries"]["teacher_vs_original_cartridge"]["oracle_correct"] == 625
    assert result["bound_is_posthoc_answer_aware"] and result["new_model_calls"] == 0


def test_bound_rejects_inconsistent_paired_counts_or_inference_labels():
    snapshot = json.loads(SNAPSHOT.read_text())
    bad = deepcopy(snapshot)
    bad["models"]["qwen"]["combined"]["pairs"]["sd_teacher_format"]["new_correct"] += 1
    with pytest.raises(ValueError, match="paired outcomes"):
        analyze(bad)
    bad = deepcopy(snapshot)
    bad["evaluation_answers_used_only_posthoc"] = False
    with pytest.raises(ValueError, match="protocol changed"):
        analyze(bad)
