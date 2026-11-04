import pytest

from pathlib import Path

from experiments.gqa_head_gain.monitor import combine, promoted_models, render_combined, monitor_units
from experiments.gqa_head_gain.report import summarize


def cases(sources):
    return [f"babilong-{length}-{source}" for source in sources for length in (8, 16, 32, 64)]


def rows(keys, improved=True):
    output = []
    for index, key in enumerate(keys):
        value = index % 100
        output.append(dict(case=key, task="qa1", length=8192, whole=int(value < 38),
            zero=int(value < 39), teacher=int(value < 40), cartridge=int(value < (45 if improved else 39)),
            baseline=None, cartridge_percent=15.))
    return output


def report(model_rows, expected):
    models = {}
    for model, items in model_rows.items():
        stats = summarize(items)
        accuracy = stats["accuracy"]
        eligible = len(items) == expected and all(accuracy["cartridge"] > accuracy[x]
                                                  for x in ("whole", "zero", "teacher"))
        models[model] = dict(rows=items, overall=stats, development_gate=dict(eligible=eligible))
    return dict(expected_per_model=expected, metric="answer_focused_lexical_v2", gain_head_mode="query",
                cached_scores=True, online_api_calls=0, evaluation_answers_used_only_posthoc=True,
                errors=[], invalid=[], models=models)


def test_promotion_requires_complete_correctly_computed_development_scores():
    keys = cases([f"qa1-{400000+i}" for i in range(50)])
    data = report({"qwen": rows(keys), "gpt": rows(keys, improved=False)}, 200)
    assert promoted_models(dict(models=["qwen", "gpt"], case_ids=keys), data) == ["qwen"]
    bad = report({"qwen": rows(keys[:-1])}, 200)
    with pytest.raises(RuntimeError):
        promoted_models(dict(models=["qwen"], case_ids=keys), bad)
    data["models"]["qwen"]["overall"]["accuracy"]["teacher"] = 0.
    with pytest.raises(ValueError):
        promoted_models(dict(models=["qwen", "gpt"], case_ids=keys), data)


def test_combined_report_checks_source_disjoint_1000_and_excludes_inspected():
    dev = cases([f"qa1-{400000+i}" for i in range(50)])
    hold = cases(["qa1-300023", "qa2-300037"] + [f"qa3-{500000+i}" for i in range(198)])
    full = combine(report({"qwen": rows(dev)}, 200), report({"qwen": rows(hold)}, 800))
    item = full["models"]["qwen"]
    assert item["overall"]["n"] == 1000
    assert item["holdout_excluding_viewed_examples"]["n"] == 792
    assert item["holdout_target_met"]
    assert "| qwen | 1000 |" in render_combined(full)
    with pytest.raises(ValueError):
        combine(report({"qwen": rows(dev)}, 200), report({"qwen": rows(dev + hold[:600])}, 800))


def test_monitor_unit_runs_once_per_hour_from_the_frozen_runtime():
    service, timer = monitor_units(Path("/tmp/workspace"), Path("/tmp/workspace/dev"))
    assert "/runtime/repo/experiments/gqa_head_gain/monitor.py" in service
    assert "OnActiveSec=1h" in timer and "OnUnitActiveSec=1h" in timer
