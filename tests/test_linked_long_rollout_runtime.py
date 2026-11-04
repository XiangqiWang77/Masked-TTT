from types import SimpleNamespace

import pytest

from experiments.babilong_linked_selection.long_rollout_runtime import linked_long_rollout_runner


def test_combination_validates_and_restores_single_request_hooks_on_failure():
    def original_manifest(manifest):
        assert manifest["selection_policy"] == "cached_decision_score10"
        assert manifest["keep_fraction"] == .1
        return 200, ["qwen", "gpt"]

    def original_result(result, entry):
        assert result["training"]["selected_transfer_weight"] == 0
        assert entry["key"] == "case"

    runner = SimpleNamespace(ROOT="old", validate_manifest=original_manifest, validate=original_result)
    runtime = SimpleNamespace(FrozenJSONSelector="score-only")
    paired = SimpleNamespace(original_question_controls="original")
    manifest = {"selection_policy": "decision_entity_link_v1", "entity_reserve_fraction": .25,
                "student_rollout_tokens": 32, "keep_fraction": .1}
    result = {"training": {"selected_transfer_weight": 0},
        "selection": {"selection_policy": "decision_entity_link_v1", "entity_reserve_fraction": .25},
        "standalone_cartridge": {"memory": {"kv_only_materialized": True,
            "cartridge_adapter_bytes": 0, "additional_source_kv_bytes_before_decode": 0,
            "additional_training_adapter_bytes_before_decode": 0}}}
    with pytest.raises(RuntimeError, match="stop"):
        with linked_long_rollout_runner(runner, runtime, paired, root="new",
                                       linked_selector="linked", physical_controls="physical"):
            assert runner.ROOT == "new" and runtime.FrozenJSONSelector == "linked"
            assert paired.original_question_controls == "physical"
            assert runner.validate_manifest(manifest) == (200, ["qwen", "gpt"])
            assert manifest["selection_policy"] == "decision_entity_link_v1"
            runner.validate(result, {"key": "case"})
            with pytest.raises(ValueError):
                runner.validate_manifest(dict(manifest, entity_reserve_fraction=.5))
            result["standalone_cartridge"]["memory"]["additional_source_kv_bytes_before_decode"] = 1
            with pytest.raises(ValueError):
                runner.validate(result, {"key": "case"})
            raise RuntimeError("stop")
    assert runner.ROOT == "old" and runner.validate_manifest is original_manifest
    assert runner.validate is original_result and runtime.FrozenJSONSelector == "score-only"
    assert paired.original_question_controls == "original"
