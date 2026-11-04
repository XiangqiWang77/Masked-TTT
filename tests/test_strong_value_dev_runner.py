from dataclasses import replace
from types import SimpleNamespace
import pytest

from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.value_lr_physical.strong_dev_runner import strong_value_runner
import experiments.anchored_short_verifier.paired_controls as paired_controls
from masked_ttt import MaskedTTT, Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine


def test_stronger_value_keeps_original_sd_and_physically_releases_cache():
    engine = gpt_engine()
    implementation = SimpleNamespace(ROOT="old", replace=replace,
        validate=lambda *_: None, validate_manifest=lambda value: value)
    old = (implementation.ROOT, implementation.replace, implementation.validate,
           implementation.validate_manifest, paired_controls.original_question_controls)
    with strong_value_runner(implementation, paired_controls, root="candidate", fixed_lr=.003,
                             physical_controls=kv_only_original_question_controls):
        manifest = dict(mode="development", case_count=200, models=["gpt"],
            fixed_value_learning_rate=.003, gain_head_mode="kv", steps=2,
            holdout_evaluated_at_selection=False)
        assert implementation.validate_manifest(manifest) == manifest
        engine.config = implementation.replace(engine.config, max_context=1024,
            self_question_count=0, guard_enabled=True, selected_transfer_weight=0.,
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        class Selector:
            def prepare(self, backbone, request):
                layout = prepare_layout(backbone, request)
                return PreparedRequest(request, layout, [1, len(layout.chunks)-2], dict(backend="unit"))
        request = Request(document="Old note red. Current note blue. " * 6,
                          question="What is the current color?", max_new_tokens=3)
        with paired_controls.original_question_controls():
            result = MaskedTTT(engine, Selector()).answer(request)
        implementation.validate(result, {})
        assert result["training"]["objective"] == "on-policy KL(teacher || student)"
        assert result["training"]["value_learning_rate"] == .003
        assert result["training"]["gain_learning_rate"] == .01
        assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
        assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
        assert result["standalone_cartridge"]["memory"]["kv_only_materialized"]
    assert (implementation.ROOT, implementation.replace, implementation.validate,
            implementation.validate_manifest, paired_controls.original_question_controls) == old


def test_other_rate_is_not_silently_substituted():
    with pytest.raises(ValueError, match="0.003"):
        with strong_value_runner(None, None, root="candidate", fixed_lr=.01, physical_controls=None):
            pass
