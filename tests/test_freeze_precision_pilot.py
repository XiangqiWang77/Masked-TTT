from copy import deepcopy
import json

import pytest

from experiments.precision_cartridge.benchmark_report import summarize
from experiments.precision_cartridge.freeze_completed_pilot import freeze, token_changes


def payload():
    cases = [f"babilong-{length}-qa1-{source}" for source in range(14) for length in (8, 16, 32, 64)]
    manifest = dict(case_ids=cases, case_count=56, models=["qwen", "gpt"], keep_fraction=.1,
        selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25, student_rollout_tokens=32,
        base_steps=2, selected_transfer_weight=0., reference_answers_used_in_inference=False, online_api_calls=0,
        precision_comparison="same_trained_state_bf16_fp32")
    finished = dict(all_complete=True, completed=dict(qwen=56, gpt=56))
    rows = [dict(case=case, whole=0, teacher=1, bf16_zero=0, bf16=1, fp32_zero=0, fp32=1,
                 bf16_kv_percent=14., fp32_kv_percent=21.) for case in cases]
    model = dict(rows=rows, overall=summarize(rows), groups={})
    report = dict(metric="answer_focused_lexical_v2", expected_per_model=56, cached_scores=True,
        online_api_calls=0, evaluation_answers_used_only_posthoc=True, errors=[], invalid=[],
        models=dict(qwen=deepcopy(model), gpt=deepcopy(model)))
    return manifest, finished, report, deepcopy(report)


def test_freeze_requires_matching_rescoring_and_no_training_labels():
    manifest, finished, saved, fresh = payload()
    output = freeze(manifest, finished, saved, fresh)
    assert output["cases_per_model"] == 56 and output["source_questions"] == 14
    assert output["models"]["qwen"]["overall"]["bf16_training_effect"]["new_correct"] == 56
    assert "rows" not in output["models"]["qwen"]
    fresh["models"]["qwen"]["rows"][0]["fp32"] = 0
    with pytest.raises(ValueError, match="rescoring differ"):
        freeze(manifest, finished, saved, fresh)
    manifest["reference_answers_used_in_inference"] = True
    with pytest.raises(ValueError, match="protocol changed"):
        freeze(manifest, finished, saved, saved)


def test_token_comparison_returns_only_aggregate_counts(tmp_path):
    path = tmp_path / "cases/qwen/case/result.json"
    path.parent.mkdir(parents=True)
    result = dict(standalone_cartridge=dict(generated_token_ids=[1, 2],
        original_question_controls=dict(zero=dict(tokens=[3, 4])),
        precision_paired_diagnostic=dict(precision_tokens=[1, 5], precision_zero=dict(tokens=[6, 4]))))
    path.write_text(json.dumps(result))
    assert token_changes(tmp_path, ["case"], "qwen") == dict(cases=1,
        trained_sequences_changed=1, trained_first_token_changed=0,
        zero_sequences_changed=1, zero_first_token_changed=1)
