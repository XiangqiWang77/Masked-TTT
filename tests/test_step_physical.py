import json

import pytest

import experiments.step_physical.report as reporter
from experiments.long_rollout_steps.report import summarize
from experiments.step_physical.protocol import development_choice, holdout_manifest
from test_value_lr_physical import fixture_value_development


def fixture_development():
    old, _, profile = fixture_value_development()
    manifest = dict(mode="development", case_count=200, case_ids=old["case_ids"],
        holdout_case_ids=old["holdout_case_ids"], models=["qwen", "gpt"],
        model_protocols={"gpt": "protocol.json"}, random_order_seed=1, keep_fraction=.1,
        step_budgets=[2, 4, 8], basis_source="document", selected_transfer_weight=0.,
        reference_answers_used_in_inference=False, online_api_calls=0)
    rows = [dict(case=case, task=case.split("-", 2)[2].split("-", 1)[0], length=8192,
                 whole=int(i < 100), zero=int(i < 100), teacher=int(i < 100),
                 step2=int(i < 100), step4=int(i < 101), step8=int(i < 103))
            for i, case in enumerate(manifest["case_ids"])]
    choice = dict(steps=8, reason="predeclared_development_rule", development_accuracy=51.5,
                  holdout_evaluated=False)
    report = dict(metric="answer_focused_lexical_v2", expected_per_model=200, development_only=True,
        selected_fraction=.1, step_budgets=[2, 4, 8], same_trajectory=True, cached_scores=True,
        online_api_calls=0, evaluation_answers_used_only_posthoc=True, errors=[], invalid=[],
        models={"gpt": dict(rows=rows, overall=summarize(rows), choice=choice)})
    return manifest, report, profile


def test_step_choice_uses_complete_development_and_keeps_source_disjoint_holdout():
    development, report, profile = fixture_development()
    choice = development_choice(development, report, "gpt")
    manifest = holdout_manifest(development, report, profile, "gpt")
    assert choice["steps"] == manifest["steps"] == 8
    assert manifest["fixed_value_learning_rate"] == .0001
    assert manifest["selection_policy"] == "cached_decision_score10"
    assert len(manifest["case_ids"]) == 800 and not manifest["holdout_evaluated_at_selection"]
    assert not set(manifest["case_ids"]) & set(manifest["development_case_ids"])


@pytest.mark.parametrize("defect", ["partial", "invalid", "aggregate", "choice"])
def test_incomplete_or_changed_step_development_cannot_promote(defect):
    manifest, report, _ = fixture_development()
    if defect == "partial":
        report["models"]["gpt"]["rows"].pop()
    elif defect == "invalid":
        report["invalid"].append(dict(model="gpt"))
    elif defect == "aggregate":
        report["models"]["gpt"]["overall"]["accuracy"]["step8"] += 1
    else:
        report["models"]["gpt"]["choice"]["steps"] = 4
    with pytest.raises(ValueError):
        development_choice(manifest, report, "gpt")


def test_report_scores_paired_checkpoints_and_keeps_missing_cases_pending(tmp_path, monkeypatch):
    development, report, profile = fixture_development()
    manifest = holdout_manifest(development, report, profile, "gpt")
    case = manifest["case_ids"][0]
    folder = tmp_path / "cases/gpt" / case
    folder.mkdir(parents=True)
    checkpoint = {str(n): dict(prediction=("yes" if n == 8 else "no"), tokens=[n]) for n in (2, 4, 8)}
    probe = dict(fixed_final_base_step=8, checkpoints=checkpoint, same_selected_kv=True,
        full_kv_used_at_decode=False, reference_answers_used=False, teacher_answer_in_final_prompt=False,
        diagnostic_snapshots_offloaded_to_cpu=True, diagnostic_snapshots_released_before_final_decode=True)
    output = dict(training=dict(fixed_base_steps=8), generated_token_ids=[8],
        self_judge=dict(native_prediction="no"), teacher=dict(prediction="no"),
        standalone_cartridge=dict(step_probe=probe, original_question_controls=dict(zero=dict(prediction="no")),
                                  memory=dict(standalone_cartridge_bytes=10)),
        memory=dict(full_document_kv_bytes=100), task="qa1", length=8192)
    (folder / "result.json").write_text(json.dumps(output))
    monkeypatch.setattr(reporter, "validate_result", lambda *_args, **_kwargs: None)
    result = reporter.collect(manifest, tmp_path, {case: dict(task="qa1", answers=["yes"])},
                              lambda text, *_args: (int(text == "yes"), True))
    assert result["overall"]["n"] == 1 and result["expected"] == 800
    assert result["overall"]["accuracy"]["step8"] == 100
    assert result["overall"]["step8_vs_step2"]["new_correct"] == 1
    assert result["strict_target_passed"] is None and not result["errors"] and not result["invalid"]
    output["standalone_cartridge"]["step_probe"]["diagnostic_snapshots_released_before_final_decode"] = False
    (folder / "result.json").write_text(json.dumps(output))
    invalid = reporter.collect(manifest, tmp_path, {case: dict(task="qa1", answers=["yes"])},
                               lambda text, *_args: (int(text == "yes"), True))
    assert invalid["overall"]["n"] == 0 and len(invalid["invalid"]) == 1
