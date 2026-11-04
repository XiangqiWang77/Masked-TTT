import pytest

from experiments.babilong_linked_selection.gpt_generic_format_report import evaluate, freeze_development


def cases(sources):
    return [f"babilong-{length}-{source}" for length in ("8", "16", "32", "64") for source in sources]


def row(case, index):
    return dict(case=case, task="qa3", length=8192, baseline_source="full_1000",
        whole=int(index < 107), baseline10=int(index < 105), zero=int(index < 122),
        teacher=int(index < 124), cartridge=int(index < 122),
        zero_teacher_format=int(index < 124), cartridge_teacher_format=int(index < 125))


def fixtures():
    dev = cases([f"qa3-{i}" for i in range(50)])
    held = cases(["qa1-300023", "qa2-300037"] + [f"qa5-{i}" for i in range(198)])
    common = dict(keep_fraction=.1, selection_policy="decision_entity_link_v1",
        entity_reserve_fraction=.25, basis_source="document", selected_transfer_weight=0,
        online_api_calls=0, reference_answers_used_in_inference=False)
    manifest = dict(common, mode="development", case_ids=dev, holdout_case_ids=held,
                    qualitative_case_sources=["qa1-300023", "qa2-300037"])
    report = dict(metric="answer_focused_lexical_v2", online_api_calls=0, cached_scores=True,
        evaluation_answers_used_only_posthoc=True, selection_policy="decision_entity_link_v1",
        development_only=True, errors=[], invalid=[], models={"gpt": {"rows": [row(k, i) for i, k in enumerate(dev)]}})
    holdout_manifest = dict(common, mode="holdout", case_ids=held)
    holdout_report = dict(report, development_only=False, expected_per_model=800, updated=1,
        models={"gpt": {"rows": [row(k, i) for i, k in enumerate(held[:4])]}})
    return manifest, report, holdout_manifest, holdout_report


def test_dev_locked_choice_and_disjoint_partial_holdout():
    manifest, dev, held_manifest, held_report = fixtures()
    decision = freeze_development(manifest, dev)
    assert decision["development"]["accuracy"]["cartridge_teacher_format"] == 62.5
    assert decision["development"]["versus_teacher"]["delta_pp"] == .5
    result = evaluate({"decision": decision}, manifest, dev, held_manifest, held_report)
    assert result["partial"] and result["holdout"]["n"] == 4 and result["combined"] is None
    assert result["generic_answer_is_pre_release_diagnostic"]
    assert result["previously_viewed_holdout_sources"] == 2


def test_does_not_select_by_holdout_and_rejects_duplicate_case():
    manifest, dev, held_manifest, held_report = fixtures()
    decision = freeze_development(manifest, dev)
    held_report["models"]["gpt"]["rows"].append(held_report["models"]["gpt"]["rows"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate({"decision": decision}, manifest, dev, held_manifest, held_report)


def test_only_rejects_errors_in_the_gpt_cohort():
    manifest, dev, held_manifest, held_report = fixtures()
    decision = freeze_development(manifest, dev)
    held_report["errors"] = [{"model": "qwen", "type": "failure"}]
    assert evaluate({"decision": decision}, manifest, dev, held_manifest, held_report)["holdout"]["n"] == 4
    held_report["errors"].append({"model": "gpt", "type": "failure"})
    with pytest.raises(ValueError, match="GPT result"):
        evaluate({"decision": decision}, manifest, dev, held_manifest, held_report)
