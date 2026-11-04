from copy import deepcopy
import json

from experiments.self_study_prompt import study_report


def candidate(baseline, final):
    flags = {key: True for key in ("same_loaded_model", "same_gpu", "same_request",
        "independently_prefilled_caches", "same_selection", "same_teacher_tokens", "same_zero_tokens",
        "baseline_standalone_released", "candidate_standalone_released")}
    pair = dict(**flags, baseline_prediction=baseline, baseline_verified=0, candidate_verified=0,
        baseline_training_steps=2, candidate_training_steps=2, same_practice_checks=True,
        baseline_native_prediction="bad", reference_answers_used=False,
        teacher_probability_tensors_compared=False)
    return dict(length=8192, prediction=final, study_prompt_pair=pair,
        training=dict(gain_learning_rate=.01, value_learning_rate=.0001, adaptive_max_answer_steps=8,
                      student_question_matches_teacher=False, steps=[{}, {}]),
        self_study=dict(verified=0), self_judge=dict(native_prediction="bad"),
        teacher=dict(prediction="good"), memory=dict(full_document_kv_bytes=100),
        standalone_cartridge=dict(original_question_controls=dict(zero=dict(prediction="bad")),
                                 memory=dict(standalone_cartridge_bytes=20)))


def test_scores_same_request_outcomes_and_rejects_invalid_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(study_report, "validate_partition", lambda *_: None)
    monkeypatch.setattr(study_report, "validate_candidate", lambda *_args, **_kwargs: None)
    cases = ["babilong-8-qa1-" + str(i) for i in range(200)]
    manifest = dict(mode="development", models=["qwen"], case_count=200, case_ids=cases,
        development_case_ids=cases, holdout_case_ids=["held"], selection_policy="decision_entity_link_v1",
        entity_reserve_fraction=.25, keep_fraction=.1, self_study_prompt_comparison="original_vs_study_only",
        student_rollout_tokens=8, selected_transfer_weight=0., reference_answers_used_in_inference=False,
        online_api_calls=0, holdout_evaluated_at_selection=False)
    values = [candidate("bad", "good"), candidate("good", "bad"), candidate("bad", "good")]
    values[2]["study_prompt_pair"]["same_gpu"] = False
    for case, value in zip(cases, values):
        path = tmp_path / "cases/qwen" / case / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    refs = {case: dict(task="qa1", answers=["good"]) for case in cases[:3]}
    score = lambda text, *_: (int(text == "good"), True)
    report = study_report.collect(manifest, tmp_path, refs, score)
    assert report["overall"]["n"] == 2 and len(report["invalid"]) == 1
    paired = report["overall"]["strictly_paired"]
    assert paired["n"] == 2 and paired["new_correct"] == paired["new_wrong"] == 1
    assert paired["delta_pp"] == 0
    assert "1 new correct / 1 new wrong" in study_report.render(report)
