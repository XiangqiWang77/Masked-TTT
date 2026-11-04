from dataclasses import replace
from types import SimpleNamespace

from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.prompt_alignment.runner import aligned_prompt_runner
import experiments.anchored_short_verifier.paired_controls as paired_controls
import masked_ttt as runtime
from masked_ttt import MaskedTTT, Request
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_public_linked_selection import _source
from test_standalone import make_engine


def test_runner_preserves_original_sd_and_restores_hooks(tmp_path):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine()
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda value: value,
        validate=lambda _result, _entry: None, replace=replace,
        preflight=lambda _engine: dict(passed=True))
    old = (implementation.ROOT, implementation.validate_manifest, implementation.validate,
           implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
           paired_controls.original_question_controls)
    with aligned_prompt_runner(implementation, runtime, paired_controls, root="candidate",
                               linked_selector=LinkedFrozenSelector,
                               physical_controls=kv_only_original_question_controls):
        manifest = dict(selection_policy="decision_entity_link_v1", entity_reserve_fraction=.25,
                        student_rollout_tokens=32)
        assert implementation.validate_manifest(manifest)["selection_policy"] == "cached_decision_score10"
        engine.config = implementation.replace(engine.config, max_context=2048, keep_fraction=.1,
            student_rollout_tokens=32, self_question_count=0, guard_enabled=False,
            selected_transfer_weight=0., value_basis_source="document",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        assert implementation.preflight(engine)["aligned_main_question"]
        selector = runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with paired_controls.original_question_controls():
            result = MaskedTTT(engine, selector).answer(Request(
                document=document, question=question, max_new_tokens=8))
        implementation.validate(result, {"key": "case"})
        assert result["training"]["student_question_matches_teacher"]
        assert result["training"]["objective"] == "on-policy KL(teacher || student)"
        assert result["training"]["selected_transfer_weight"] == 0
        assert len(result["training"]["steps"]) == 2
        assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
        assert result["standalone_cartridge"]["memory"]["kv_only_materialized"]
    assert (implementation.ROOT, implementation.validate_manifest, implementation.validate,
            implementation.replace, implementation.preflight, runtime.FrozenJSONSelector,
            paired_controls.original_question_controls) == old
