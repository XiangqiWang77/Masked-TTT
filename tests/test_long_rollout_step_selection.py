from copy import deepcopy

import pytest

from experiments.long_rollout_steps.report import summarize
from experiments.long_rollout_steps.selection import development_choices
from test_long_rollout_steps_extend import fixture_protocol


def fixture_development():
    pilot, _ = fixture_protocol()
    manifest = dict(pilot, case_count=200, case_ids=pilot["development_case_ids"])
    rows = [dict(case=case, task=case.split("-", 3)[2], length=int(case.split("-")[1]),
                 whole=0, zero=0, teacher=0, step2=0, step4=0, step8=0)
            for case in manifest["case_ids"]]
    for row in rows[:20]:
        row.update(whole=1, zero=1, teacher=1, step2=1, step4=1, step8=1)
    for row in rows[20:23]:
        row.update(step2=1, step4=1, step8=1)
    report = dict(expected_per_model=200, metric="answer_focused_lexical_v2", cached_scores=True,
        online_api_calls=0, evaluation_answers_used_only_posthoc=True, errors=[], invalid=[],
        models={model: {"rows": deepcopy(rows), "overall": summarize(rows)} for model in manifest["models"]})
    return manifest, report


def test_ties_choose_fewer_regressions_then_fewer_steps():
    manifest, report = fixture_development()
    choices = development_choices(manifest, report)
    assert all(choice["eligible"] and choice["fixed_final_base_step"] == 2 for choice in choices.values())
    assert choices["qwen"]["delta_pp"]["teacher"] == 1.5


def test_best_step_is_fixed_separately_per_model():
    manifest, report = fixture_development()
    rows = report["models"]["gpt"]["rows"]
    rows[23]["step4"] = 1
    report["models"]["gpt"]["overall"] = summarize(rows)
    choices = development_choices(manifest, report)
    assert choices["qwen"]["fixed_final_base_step"] == 2
    assert choices["gpt"]["fixed_final_base_step"] == 4


def test_no_holdout_when_best_checkpoint_fails_a_control():
    manifest, report = fixture_development()
    for model in report["models"]:
        for row in report["models"][model]["rows"]:
            row["teacher"] = row["step2"]
        report["models"][model]["overall"] = summarize(report["models"][model]["rows"])
    assert all(not choice["eligible"] for choice in development_choices(manifest, report).values())


def test_unmatched_or_stale_development_cannot_select_a_step():
    manifest, report = fixture_development()
    report["models"]["qwen"]["rows"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        development_choices(manifest, report)
    manifest, report = fixture_development()
    report["models"]["qwen"]["rows"][0]["step4"] = 0
    with pytest.raises(ValueError, match="statistics"):
        development_choices(manifest, report)
