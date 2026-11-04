import json

import pytest

from experiments.babilong_linked_selection.long_rollout_protocol import holdout_decision, summarize
from experiments.long_rollout_steps.holdout_execute import model_paths
from experiments.long_rollout_steps.holdout_report import combine, collect, render
from experiments.long_rollout_steps.selection import development_choices
from test_long_rollout_step_selection import fixture_development


def fixture_holdout(tmp_path):
    dev_manifest, dev_report = fixture_development()
    choices = development_choices(dev_manifest, dev_report)
    steps = {model: item["fixed_final_base_step"] for model, item in choices.items() if item["eligible"]}
    protocols = {}
    for model in steps:
        path = tmp_path / (model + "-protocol.json")
        path.write_text(json.dumps({"model_path": str(tmp_path / (model + "-model"))}))
        protocols[model] = str(path)
    manifest = dict(dev_manifest, mode="holdout", case_count=800,
        case_ids=list(dev_manifest["holdout_case_ids"]), models=list(steps),
        fixed_final_base_steps=steps, model_protocols=protocols,
        entity_reserve_fraction=.25, qualitative_case_sources=["qa1-300023", "qa2-300037"],
        holdout_evaluated_at_selection=False)
    models = {}
    for model in steps:
        rows = [dict(case=case, task=case.split("-", 3)[2], length=int(case.split("-")[1]),
            whole=int(i < 100), zero=int(i < 100), teacher=int(i < 100),
            cartridge=int(i < 140), linked8=None, cartridge_percent=10.)
            for i, case in enumerate(manifest["case_ids"])]
        strict = [row for row in rows if row["case"].split("-", 2)[2]
                  not in {"qa1-300023", "qa2-300037"}]
        models[model] = dict(rows=rows, overall=summarize(rows),
            excluding_viewed_examples=summarize(strict), decision=holdout_decision(summarize(strict)))
    report = dict(mode="holdout", expected_per_model=800, metric="answer_focused_lexical_v2",
        selection_policy="decision_entity_link_v1", student_rollout_tokens=32,
        selected_transfer_weight=0., cached_scores=True, online_api_calls=0,
        evaluation_answers_used_only_posthoc=True, fixed_final_base_steps=steps,
        models=models, errors=[], invalid=[])
    return dev_manifest, dev_report, manifest, report


def test_model_paths_are_fixed_before_any_holdout_generation(tmp_path):
    _, _, manifest, _ = fixture_holdout(tmp_path)
    assert set(model_paths(manifest).values()) == {2}
    manifest["fixed_final_base_steps"]["qwen"] = 3
    with pytest.raises(ValueError, match="development-fixed"):
        model_paths(manifest)


def test_empty_holdout_report_remains_partial(tmp_path):
    _, _, manifest, _ = fixture_holdout(tmp_path)
    report = collect(manifest, tmp_path, {}, lambda *args: (0, True))
    assert report["expected_per_model"] == 800
    assert all(item["overall"]["n"] == 0 for item in report["models"].values())


def test_combine_200_and_800_without_selecting_on_holdout(tmp_path):
    dev_manifest, dev, manifest, holdout = fixture_holdout(tmp_path)
    result = combine(dev_manifest, dev, manifest, holdout)
    assert result["mode"] == "combined" and result["expected_per_model"] == 1000
    for model in manifest["models"]:
        assert result["models"][model]["overall"]["n"] == 1000
        assert result["models"][model]["holdout"]["n"] == 800
        assert result["models"][model]["holdout_excluding_viewed_examples"]["n"] == 792
        assert result["fixed_final_base_steps"][model] == 2
    note = render(result)
    assert "entity-linked context" in note and "32 tokens" in note
    assert "rather than an exact configuration reproduction" in note


def test_combine_rejects_a_holdout_selected_on_its_own_scores(tmp_path):
    dev_manifest, dev, manifest, holdout = fixture_holdout(tmp_path)
    manifest["fixed_final_base_steps"]["qwen"] = 4
    holdout["fixed_final_base_steps"]["qwen"] = 4
    with pytest.raises(ValueError, match="not selected"):
        combine(dev_manifest, dev, manifest, holdout)
