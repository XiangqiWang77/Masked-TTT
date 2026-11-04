from copy import deepcopy

import pytest

from experiments.babilong_linked_selection.validation_protocol import build_manifest


def _development_and_results():
    lengths = (8, 16, 32, 64)
    dev_sources = [f"qa1-{400000 + i}" for i in range(50)]
    hold_sources = ["qa1-300023", "qa2-300037"] + [f"qa2-{500000 + i}" for i in range(198)]
    dev = [f"babilong-{length}-{source}" for source in dev_sources for length in lengths]
    hold = [f"babilong-{length}-{source}" for source in hold_sources for length in lengths]
    development = dict(
        case_ids=dev, holdout_case_ids=hold, keep_fraction=.1,
        selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25,
        selected_transfer_weight=0., qualitative_case_sources=["qa1-300023", "qa2-300037"],
        model_protocols={}, random_order_seed=42,
    )

    def scored(candidate_count, teacher_count):
        rows = [dict(
            case=key, whole=int(i < 100), zero=int(i < 110),
            teacher=int(i < teacher_count), cartridge=int(i < candidate_count),
            baseline10=int(i < 106), baseline_source="full_1000",
        ) for i, key in enumerate(dev)]
        rate = lambda name: 100 * sum(row[name] for row in rows) / len(rows)
        accuracy = {key: rate(key) for key in ("whole", "zero", "teacher", "cartridge")}
        return dict(
            rows=rows,
            overall=dict(n=200, accuracy=accuracy, versus_teacher={}, versus_zero={}),
            matched_10=dict(n=200, candidate_accuracy=accuracy["cartridge"],
                            baseline10_accuracy=rate("baseline10"), delta_pp=0.),
        )

    latest = dict(
        metric="answer_focused_lexical_v2", development_only=True,
        expected_per_model=200, selected_fraction=.1,
        selection_policy="decision_entity_link_v1", errors=[], invalid=[],
        models=dict(qwen=scored(120, 125), gpt=scored(104, 104)),
    )
    return development, latest


def test_selection_validation_does_not_claim_strict_teacher_success():
    development, latest = _development_and_results()
    manifest = build_manifest(development, latest)
    qwen = manifest["development_decisions"]["qwen"]
    assert manifest["models"] == ["qwen"]
    assert qwen["selection_validation_eligible"] and not qwen["strict_olmo_target_eligible"]
    assert manifest["selection_rule_added_after_development_review"]
    assert manifest["case_count"] == 800 and len(manifest["case_ids"]) == 800
    assert not manifest["holdout_evaluated_at_selection"]


def test_missing_pairs_and_source_leakage_block_promotion():
    development, latest = _development_and_results()
    latest["models"]["qwen"]["matched_10"]["n"] = 199
    with pytest.raises(RuntimeError, match="Waiting"):
        build_manifest(development, latest)
    development, latest = _development_and_results()
    development["holdout_case_ids"][0] = development["case_ids"][0]
    with pytest.raises(ValueError, match="split"):
        build_manifest(development, latest)


def test_no_selection_promotion_if_update_reduces_selected_zero_accuracy():
    development, latest = _development_and_results()
    changed = deepcopy(latest)
    rows = changed["models"]["qwen"]["rows"]
    for index, row in enumerate(rows):
        row["zero"] = int(index < 121)
    changed["models"]["qwen"]["overall"]["accuracy"]["zero"] = 60.5
    manifest = build_manifest(development, changed)
    assert not manifest["development_decisions"]["qwen"]["selection_validation_eligible"]
    assert "qwen" not in manifest["models"]
