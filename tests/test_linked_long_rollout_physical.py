"""Integration of cached linked evidence, a long on-policy rollout, and KV-only decode."""
from dataclasses import replace

import pytest

from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from masked_ttt import MaskedTTT, Request
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_public_linked_selection import _source
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_long_on_policy_rollout_has_physical_independent_final_cache(architecture, tmp_path):
    document, question, _, _ = _source(tmp_path)
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    engine.config = replace(engine.config, keep_fraction=.1, student_rollout_tokens=32,
        self_question_count=0, guard_enabled=False, selected_transfer_weight=0.,
        value_basis_source="document", release_full_before_cartridge=True,
        release_selected_source_before_decode=True,
        teacher_response_instruction="Return only the final answer.")
    # A tiny random model may generate EOS before 32. This test fixes the
    # rollout length; production model EOS behavior is unchanged.
    engine.eos_ids = {-1}
    request = Request(document=document, question=question, max_new_tokens=32)
    selector = LinkedFrozenSelector(tmp_path)
    selector.bind_case("case")
    with kv_only_original_question_controls():
        result = MaskedTTT(engine, selector).answer(request)
    first = result["training"]["steps"][0]
    standalone = result["standalone_cartridge"]
    controls, memory = standalone["original_question_controls"], standalone["memory"]
    assert result["selection"]["selected_chunks"] == [1, 2, 3, 6]
    assert result["training"]["objective"] == "on-policy KL(teacher || student)"
    assert len(first["student_rollout_tokens"]) == 32
    assert first["target_positions"] == list(range(32))
    assert first["gain_gradient_norm"] > 0 and first["value_gradient_norm"] > 0
    assert result["training"]["selected_transfer_weight"] == 0
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    assert standalone["generated_token_ids"] == controls["adapted"]["tokens"]
    assert controls["same_question_suffix"] and controls["same_selected_kv"]
    assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0
    assert memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
