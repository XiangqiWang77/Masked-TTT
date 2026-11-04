import pytest

from experiments.babilong_linked_selection.freeze_three_model_table import collect


def inputs():
    overall = dict(n=1000, whole_accuracy=36.1, teacher_accuracy=41.2,
                   cartridge_accuracy=44.6, vs_teacher={"delta_points": 3.4})
    original = dict(metric="answer_focused_lexical_v2", source_questions_per_model=250,
                    models={name: {"overall": dict(overall)} for name in ("olmo", "qwen", "gpt")})
    row = dict(n=1000, accuracy=dict(whole=50, teacher=60, zero=59, cartridge=61,
        zero_teacher_format=60, cartridge_teacher_format=61), pairs={
        "teacher_to_cartridge": {"delta_pp": 1}, "sd_original_format": {"delta_pp": 2},
        "selection_and_method": {"delta_pp": 3}})
    linked = dict(metric="answer_focused_lexical_v2", selected_fraction=.1,
                  models={name: dict(state="complete", combined=row) for name in ("qwen", "gpt")})
    return original, linked


def test_table_keeps_unsaved_olmo_zero_missing():
    original, linked = inputs()
    result = collect(original, linked)
    assert result["original_selection"]["olmo"]["zero"] is None
    assert result["linked_selection"]["gpt"]["vs_teacher_pp"] == 1


def test_rejects_different_scorer_and_incomplete_holdout():
    original, linked = inputs()
    linked["metric"] = "different"
    with pytest.raises(ValueError):
        collect(original, linked)
    original, linked = inputs()
    linked["models"]["gpt"]["state"] = "in_progress"
    with pytest.raises(ValueError):
        collect(original, linked)
