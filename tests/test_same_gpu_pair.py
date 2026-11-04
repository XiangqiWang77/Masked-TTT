from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from experiments.prompt_alignment.same_gpu_pair import run_pair, validate_configs, validate_pair
from masked_ttt import Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


class Selector:
    def prepare(self, backbone, request):
        layout = prepare_layout(backbone, request)
        indices = [1, len(layout.chunks)-2]
        metadata = dict(backend="unit", selected_original_chunks=indices, selected_chunks=indices,
                        total_chunks=len(layout.chunks), selected_tokens=sum(len(layout.chunks[i]) for i in indices))
        return PreparedRequest(request, layout, indices, metadata)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
def test_same_device_pair_runs_original_sd_and_releases_both_caches(architecture):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    left = replace(engine.config, steps=2, self_question_count=0, guard_enabled=False,
        selected_transfer_weight=0, value_basis_source="document", standalone_answer_format="original",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    right = replace(left, align_student_teacher_prompt=True)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=3)
    before = torch.get_rng_state().clone()
    original_config = engine.config
    result = run_pair(engine, Selector(), request, left, right)
    assert torch.equal(before, torch.get_rng_state())
    assert engine.config is original_config
    assert result["changed_settings"] == ["align_student_teacher_prompt"]
    assert result["same_loaded_model"] and result["independently_prefilled_caches"]
    assert result["same_device"] and not result["same_gpu"] and result["device_type"] == "cpu"
    assert not result["teacher_probability_tensors_compared"]
    for value in (result["left"], result["right"]):
        assert value["standalone_cartridge"]["memory"]["kv_only_materialized"]
        assert not value["teacher_answer_in_final_prompt"]
    broken = deepcopy(result["right"])
    broken["teacher"]["tokens"] = [999]
    with pytest.raises(ValueError, match="Teacher output"):
        validate_pair(result["left"], broken)


def test_no_confounded_teacher_or_training_objective_changes():
    config = make_engine().config
    with pytest.raises(ValueError, match="optimization settings"):
        validate_configs(config, replace(config, teacher_response_instruction="Different teacher"))
    with pytest.raises(ValueError, match="original standalone"):
        guarded = replace(config, guard_enabled=True)
        validate_configs(guarded, guarded)
