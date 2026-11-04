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
