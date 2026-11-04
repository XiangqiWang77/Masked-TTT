"""Verify both answer suffixes on an already-loaded real backbone.

The synthetic request uses no benchmark answers or API calls. This optional
preflight runs the same OLMo-style SD and checks that a trained, independently
materialized selected cache produces the generic-format control again after
full KV, selected source KV, and training state are released.
"""
from dataclasses import replace

import torch

from experiments.kv_only_cartridge.paired_control import physical_paired_prompt_control
from masked_ttt import MaskedTTT, Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest


def validate_loaded_backbone_formats(engine):
    old_config = engine.config
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if engine.device.type == "cuda" else None
    try:
        engine.config = replace(old_config, chunk_tokens=64, steps=2, max_answer_steps=2,
            self_question_count=0, student_rollout_tokens=3, guard_enabled=False,
            standalone_cartridge_enabled=True, standalone_answer_format="original",
            selected_transfer_weight=0., value_basis_source="document",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        request = Request(document="The old note is red. " + "Ordinary context. " * 30 +
                          "The current note is blue.",
                          question="What color is the current note?", max_new_tokens=4)

        class Selector:
            def prepare(self, backbone, item):
                layout = prepare_layout(backbone, item)
                return PreparedRequest(item, layout, [len(layout.chunks) // 2],
                                       dict(backend="synthetic_format_release_check"))

        with physical_paired_prompt_control(verify_teacher_suffix_after_release=True):
            result = MaskedTTT(engine, Selector()).answer(request, time_budget_seconds=300)
        memory = result["standalone_cartridge"]["memory"]
        controls = result["standalone_cartridge"]["paired_prompt_control"]
        verified = controls["teacher_suffix_independent_decode"]
        if (not result["backbone_unchanged"] or not result["original_document_KV_unchanged"] or
            result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0 or
            memory["additional_source_kv_bytes_before_decode"] != 0 or
            memory["additional_training_adapter_bytes_before_decode"] != 0 or
            memory["cartridge_adapter_bytes"] != 0 or
            not verified["source_and_training_released"] or verified["adapter_bytes"] != 0 or
            verified["teacher_answer_in_prompt"] or
            verified["generated_token_ids"] != controls["trained_teacher_suffix"]["tokens"]):
            raise AssertionError("Both formats require a physically independent cartridge")
        return dict(passed=True, original_and_generic_same_materialized_kv=True,
            generic_tokens_match_pre_release=True, source_and_training_released=True,
            adapter_bytes=0, teacher_answer_in_prompt=False, online_api_calls=0,
            reference_answers_used=False, model_type=engine.model.config.model_type)
    finally:
        engine.config = old_config
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
