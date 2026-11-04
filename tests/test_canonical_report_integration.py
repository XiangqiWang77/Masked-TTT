"""Exercise the paired runtime's real result schema through its report."""
import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

import masked_ttt as runtime
import experiments.anchored_short_verifier.paired_controls as controls
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.value_lr_physical.canonical_job import make_manifest
from experiments.value_lr_physical.canonical_report import collect
from experiments.value_lr_physical.canonical_runner import canonical_pair_runner
from masked_ttt import Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_canonical_job import sources
from test_canonical_pair import BoundaryTokenizer
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize("model", ["qwen", "gpt"])
def test_real_small_model_result_roundtrips_through_post_generation_report(tmp_path, monkeypatch, model):
    monkeypatch.setenv("MASKED_TTT_FIXED_CHAT_DATE", "2026-11-03")
    engine = make_engine() if model == "qwen" else gpt_engine()
    if model == "qwen":
        engine.tokenizer = BoundaryTokenizer()
    else:
        class HarmonyBoundaryTokenizer(BoundaryTokenizer):
            def apply_chat_template(self, messages, **_kwargs):
                return "<|start|>user<|message|>" + messages[0]["content"] + "<|end|><|start|>assistant"
        from masked_ttt.architectures import FinalChannelTokenizer
        engine.tokenizer = FinalChannelTokenizer(HarmonyBoundaryTokenizer())
    manifest = make_manifest(*sources())
    key = manifest["case_ids"][0]

    class Selector:
        def prepare(self, backbone, request):
            layout = prepare_layout(backbone, request)
            selected = [2]
            metadata = dict(backend="precomputed_source_JSON", source_case_key=key,
                selected_original_chunks=[[2]], selected_chunks=selected, total_chunks=len(layout.chunks),
                selected_tokens=len(layout.chunks[2]), selection_policy="decision_entity_link_v1",
                entity_reserve_fraction=.25, complete_original_document=True,
                reference_answers_used=False, online_api_calls=0)
            return PreparedRequest(request, layout, selected, metadata)

    impl = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x, validate=lambda *_: None,
                           replace=replace, preflight=lambda _: {"passed": True})
    with canonical_pair_runner(impl, runtime, controls, root="candidate", selector=Selector,
                               physical_controls=kv_only_original_question_controls):
        engine.config = impl.replace(engine.config, max_context=engine.config.max_context,
            chunk_tokens=32, self_question_count=0, guard_enabled=True, selected_transfer_weight=0.,
            value_basis_source="document", release_full_before_cartridge=True,
            release_selected_source_before_decode=True, student_rollout_tokens=8)
        with controls.original_question_controls():
            result = runtime.MaskedTTT(engine, Selector()).answer(
                Request(document="A" * 32 + "B" * 32 + "E" * 32 + "F" * 32,
                        question="Which block is E?", max_new_tokens=3))
        # This is a schema-only CPU test. A real job still must pass its
        # separate loaded-model GPU preflight; this flag cannot do that.
        assert not result["canonical_tokenization_pair"]["same_gpu"]
        result["canonical_tokenization_pair"]["same_gpu"] = True
        impl.validate(result, {"key": key})

    result.update(key=key, model_short=model, task="qa1", length=8192, keep_fraction=.1)
    folder = tmp_path / "cases" / model / key
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps(result))
    report = collect(manifest, tmp_path, {key: dict(task="qa1", answers=["fixture"])},
                     lambda *_args: (0, None))
    assert not report["errors"] and not report["invalid"]
    assert report["models"][model]["overall"]["n"] == 1
    assert report["models"][model]["rows"][0]["case"] == key
    assert report["reference_answers_used_only_posthoc"] and report["online_api_calls"] == 0
