from copy import deepcopy

from experiments.self_study_prompt.study_gate import development_decision, probe_decision


def test_probe_requires_completed_label_free_paired_signal():
    report = dict(state="complete", n=10, expected=10, errors=0, reference_answers_used=False,
        online_api_calls=0, training_runtime_changed=False, same_selected_teacher_cache=True,
        downstream_accuracy_not_measured=True, base=dict(n=10, verified_exercises=1),
        variant=dict(n=10, verified_exercises=3), more_verified=2, fewer_verified=0)
    assert probe_decision(report)["eligible"]
    for changes in (dict(state="partial"), dict(reference_answers_used=True), dict(online_api_calls=1),
                    dict(more_verified=1, fewer_verified=1), dict(variant=dict(n=10, verified_exercises=1))):
        assert not probe_decision(dict(report, **changes))["eligible"]


def test_full_development_requires_better_same_run_and_original_sd():
    report = dict(mode="development", expected=200, metric="answer_focused_lexical_v2", errors=[], invalid=[],
        overall=dict(n=200, accuracy=dict(whole=50., zero=52., teacher=54., cartridge=56.),
                     strictly_paired=dict(n=200, delta_pp=1., new_correct=3, new_wrong=1)))
    assert development_decision(report)["eligible"]
    for changed in ("teacher", "zero", "whole"):
        copy = deepcopy(report)
        copy["overall"]["accuracy"][changed] = 56.
        assert not development_decision(copy)["eligible"]
    copy = deepcopy(report)
    copy["overall"]["strictly_paired"]["delta_pp"] = 0.
    assert not development_decision(copy)["eligible"]
    copy = deepcopy(report)
    copy["overall"]["strictly_paired"]["n"] = 199
    assert not development_decision(copy)["eligible"]
