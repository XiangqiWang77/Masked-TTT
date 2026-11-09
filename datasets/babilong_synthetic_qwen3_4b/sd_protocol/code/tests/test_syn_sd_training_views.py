from experiments.run_syn_sd_factorial import arm_training_config, training_views


def fixture():
    targets = {"data": {"train": ["all-train"], "validation": ["all-val"]}}
    selected = {"data": {"train": [], "validation": ["selected-val"]}}
    return targets, selected


def test_all_rejected_selective_sd_falls_back_on_train_and_validation():
    targets, selected = fixture()
    views = training_views(
        targets,
        selected,
        "data",
        "selective_sd",
        {
            "empty_selection_policy": "original_SD",
            "selective_validation_policy": "independently_admitted_teacher_targets",
            "selective_consolidation_schedule": False,
        },
    )
    assert views["values"] is targets["data"]["train"]
    assert views["heldout"] is targets["data"]["validation"]
    assert views["fallback"] is True
    assert views["validation_fallback"] is True


def test_direct_selective_sd_uses_only_admitted_signals_without_repetition():
    targets = {"data": {"train": ["a", "b", "c"], "validation": ["v1", "v2"]}}
    selected = {"data": {"train": ["a", "c"], "validation": ["v2"]}}
    views = training_views(
        targets,
        selected,
        "data",
        "selective_sd",
        {
            "empty_selection_policy": "original_SD",
            "selective_validation_policy": "independently_admitted_teacher_targets",
            "selective_consolidation_schedule": False,
        },
    )
    assert views["values"] == ["a", "c"]
    assert views["heldout"] == ["v2"]
    assert views["consolidation"] is None
    assert views["fallback"] is False


def test_vanilla_sd_ignores_selective_views():
    targets, selected = fixture()
    views = training_views(targets, selected, "data", "vanilla_sd", {})
    assert views["values"] is targets["data"]["train"]
    assert views["heldout"] is targets["data"]["validation"]
    assert views["fallback"] is False


def test_tracin_selective_sd_uses_learning_dynamics_without_heldout():
    targets = {"data": {"train": ["all"], "validation": ["heldout"]}}
    selected = {"data": {"train": ["positive"], "validation": ["unused"]}}
    views = training_views(
        targets,
        selected,
        "data",
        "selective_sd",
        {
            "selective_consolidation_schedule": True,
            "selective_validation_policy": "no_heldout_tracin_training_dynamics",
            "empty_selection_policy": "original_SD",
        },
    )
    assert views["values"] == ["all"]
    assert views["consolidation"] == ["positive"]
    assert views["heldout"] == []
    assert views["fallback"] is False


def test_packed_update_matching_preserves_reference_and_matches_other_arms():
    config = {
        "matched_update_reference_arm": "vanilla_syns_vanilla_sd",
        "matched_update_unit": "packed_optimizer_steps",
        "official_epochs": 2,
        "train_batch_size": "full_dataset",
        "min_steps": 1,
        "max_steps": 2,
        "eval_every": 1,
        "patience": 3,
    }
    results = {"vanilla_syns_vanilla_sd": {"training": {"updates": 14}}}

    reference = arm_training_config(
        config, results, "vanilla_syns_vanilla_sd"
    )
    matched = arm_training_config(
        config, results, "specific_syns_vanilla_sd"
    )

    assert reference["official_epochs"] == 2
    assert reference["train_batch_size"] == "full_dataset"
    assert "fixed_updates" not in reference
    assert "official_epochs" not in matched
    assert matched["fixed_updates"] == 14
    assert matched["train_batch_size"] == 1
    assert matched["eval_every"] == 7
    assert matched["max_steps"] == 14


def test_matched_selective_preserves_first_half_for_ordinary_sd():
    config = {
        "matched_update_reference_arm": "vanilla_syns_vanilla_sd",
        "matched_update_unit": "packed_optimizer_steps",
        "official_epochs": 2,
        "train_batch_size": "full_dataset",
        "min_steps": 1,
        "max_steps": 2,
        "eval_every": 1,
        "patience": 3,
        "selective_consolidation_schedule": True,
        "selective_consolidation_start_policy": "matched_halfway",
        "selective_checkpoint_after_consolidation_only": True,
    }
    results = {"vanilla_syns_vanilla_sd": {"training": {"updates": 14}}}

    matched = arm_training_config(
        config, results, "specific_syns_selective_sd"
    )

    assert matched["fixed_updates"] == 14
    assert matched["selective_consolidation_start_step"] == 8
    assert matched["min_steps"] == 8
    assert matched["track_best_before_min"] is False
