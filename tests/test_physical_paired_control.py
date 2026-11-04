from dataclasses import replace

import pytest

from experiments.kv_only_cartridge.paired_control import physical_paired_prompt_control
from masked_ttt import MaskedTTT, Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
@pytest.mark.parametrize("compact", [False, True])
def test_independent_answer_matches_all_control_conditions(architecture, compact):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    engine.config = replace(
        engine.config, self_question_count=0, guard_enabled=False,
        selected_transfer_weight=0., value_basis_source="document",
        release_full_before_cartridge=True, release_selected_source_before_decode=True,
    )
    request = Request(
        document="The old note is red. The current note is blue. " * 6,
        question="Which color is current?", max_new_tokens=3,
    )

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    method = MaskedTTT(engine, Selector())
    reference = method.answer(request)
    with physical_paired_prompt_control(compact=compact):
        result = method.answer(request)
    standalone = result["standalone_cartridge"]
    controls, memory = standalone["paired_prompt_control"], standalone["memory"]
    assert controls["trained_original"]["tokens"] == result["generated_token_ids"]
    assert result["generated_token_ids"] == reference["generated_token_ids"]
    assert result["teacher"]["tokens"] == reference["teacher"]["tokens"]
    assert controls["same_selected_kv"] and controls["same_generation_limit"]
    assert controls["same_backbone_and_device"] and not controls["reference_answers_used"]
    assert not controls["teacher_answer_in_student_prompt"]
    assert memory["additional_source_kv_bytes_before_decode"] == 0
    assert memory["additional_training_adapter_bytes_before_decode"] == 0
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    if compact:
        assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"] == 0


def test_hooks_are_restored_after_failure():
    import masked_ttt.pipeline as pipeline

    answer, decode = pipeline.answer_standalone, pipeline.decode_standalone
    with pytest.raises(RuntimeError, match="intentional"):
        with physical_paired_prompt_control():
            raise RuntimeError("intentional")
    assert pipeline.answer_standalone is answer and pipeline.decode_standalone is decode


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_generic_suffix_decodes_from_independent_materialized_cartridge(architecture):
    from masked_ttt.kv_only import kv_only_cartridge

    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    engine.config = replace(engine.config, self_question_count=0, guard_enabled=False,
        selected_transfer_weight=0., value_basis_source="document", standalone_answer_format="original",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks) - 2], dict(backend="unit"))

    method = MaskedTTT(engine, Selector())
    with physical_paired_prompt_control(verify_teacher_suffix_after_release=True):
        result = method.answer(request)
    controls = result["standalone_cartridge"]["paired_prompt_control"]
    verification = controls["teacher_suffix_independent_decode"]
    assert verification["source_and_training_released"] and verification["adapter_bytes"] == 0
    assert not verification["teacher_answer_in_prompt"]
    assert verification["generated_token_ids"] == controls["trained_teacher_suffix"]["tokens"]
    assert result["generated_token_ids"] == controls["trained_original"]["tokens"]
    assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] == 0

    # An independently invoked generic-format cartridge follows the identical
    # request-local training and must produce the materialized alternate output.
    engine.config = replace(engine.config, standalone_answer_format="teacher_requirements")
    with kv_only_cartridge():
        direct = method.answer(request)
    assert direct["generated_token_ids"] == verification["generated_token_ids"]
    assert direct["standalone_cartridge"]["answer_format"] == "teacher_requirements"


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_format_release_preflight_restores_configuration_and_rng(architecture):
    import torch
    from experiments.kv_only_cartridge.format_release_probe import validate_loaded_backbone_formats

    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    config, state = engine.config, torch.get_rng_state()
    checked = validate_loaded_backbone_formats(engine)
    assert checked["passed"] and checked["generic_tokens_match_pre_release"]
    assert checked["direct_request_tokens_and_logits_match"]
    assert checked["source_and_training_released"] and checked["adapter_bytes"] == 0
    assert not checked["reference_answers_used"] and not checked["teacher_answer_in_prompt"]
    assert engine.config is config and torch.equal(torch.get_rng_state(), state)
