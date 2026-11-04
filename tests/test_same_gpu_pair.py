from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from experiments.prompt_alignment.same_gpu_pair import run_pair, run_study_prompt_pair, validate_configs, validate_pair
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
        transfer = replace(config, selected_transfer_weight=.1)
        validate_configs(transfer, transfer)


def test_study_only_pair_changes_only_study_prompt_and_restores_hook(monkeypatch):
    import masked_ttt.pipeline as pipeline
    engine = make_engine()
    config = replace(engine.config, steps=2, self_question_count=1, self_question_max_tokens=8,
        guard_enabled=False, selected_transfer_weight=0, value_basis_source="document",
        standalone_answer_format="original", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=3)
    calls, main_suffixes = [], []
    original = pipeline.generate_self_study
    original_main = pipeline.main_student_suffix

    def capture(*args):
        calls.append(args[3].prompt)
        return original(*args)

    def capture_main(*args, **kwargs):
        value = original_main(*args, **kwargs)
        main_suffixes.append(value.clone())
        return value

    monkeypatch.setattr(pipeline, "generate_self_study", capture)
    monkeypatch.setattr(pipeline, "main_student_suffix", capture_main)
    result = run_study_prompt_pair(engine, Selector(), request, config)
    assert pipeline.generate_self_study is capture
    assert len(calls) == 2 and calls[0] == request.prompt
    assert "for identifying study facts only; do not answer it:" in calls[1]
    assert result["changed_settings"] == ["self_study_prompt"]
    assert result["left"]["teacher"]["tokens"] == result["right"]["teacher"]["tokens"]
    assert len(main_suffixes) == 2 and torch.equal(*main_suffixes)
    assert result["left"]["training"]["objective"] == result["right"]["training"]["objective"]
    assert not result["reference_answers_used"]
    changed = deepcopy(result["right"])
    changed["self_study"]["checks"] = [dict(accepted=True)]
    with pytest.raises(ValueError, match="practice sets differ"):
        validate_pair(result["left"], changed)
    assert not validate_pair(result["left"], changed, require_same_practices=False)


def test_pair_uses_one_shared_deadline_and_restores_rng():
    engine = make_engine()
    config = replace(engine.config, self_question_count=0, guard_enabled=False,
        selected_transfer_weight=0, value_basis_source="document", standalone_answer_format="original",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    before = torch.get_rng_state().clone()
    with pytest.raises(TimeoutError, match="paired request budget"):
        run_pair(engine, Selector(), None, config, config, time_budget_seconds=0)
    assert torch.equal(before, torch.get_rng_state())


def test_olmo_full_diagnostic_does_not_change_standalone_contract():
    config = replace(make_engine().config, guard_enabled=True, selected_transfer_weight=0,
        value_basis_source="document", standalone_answer_format="original",
        release_full_before_cartridge=True, release_selected_source_before_decode=True)
    assert validate_configs(config, config) == []


def test_study_pair_preserves_full_diagnostic_with_original_final_cartridge():
    engine = make_engine()
    config = replace(engine.config, steps=2, self_question_count=1, self_question_max_tokens=8,
        guard_enabled=True, selected_transfer_weight=0, value_basis_source="document",
        standalone_answer_format="original", release_full_before_cartridge=True,
        release_selected_source_before_decode=True)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="Which color is current?", max_new_tokens=3)
    result = run_study_prompt_pair(engine, Selector(), request, config)
    left, right = result["left"], result["right"]
    assert isinstance(left["self_judge"]["native_prediction"], str)
    assert left["self_judge"]["native_prediction"] == right["self_judge"]["native_prediction"]
    assert left["self_judge"]["applies_to"] == right["self_judge"]["applies_to"] == "full_context_diagnostic"
    assert left["output_source"] == right["output_source"] == "standalone_cartridge"
    assert left["standalone_cartridge"]["answer_format"] == right["standalone_cartridge"]["answer_format"] == "original"


def test_cannot_compare_study_prompts_when_study_disabled():
    engine = make_engine()
    config = replace(engine.config, self_question_count=0)
    with pytest.raises(ValueError, match="self-study enabled"):
        run_study_prompt_pair(engine, Selector(), None, config)


def test_study_prompt_hook_restores_after_error():
    import masked_ttt.pipeline as pipeline
    from experiments.self_study_prompt.study_context import study_only_prompt
    original = pipeline.generate_self_study
    with pytest.raises(RuntimeError, match="deliberate"):
        with study_only_prompt():
            assert pipeline.generate_self_study is not original
            raise RuntimeError("deliberate")
    assert pipeline.generate_self_study is original
