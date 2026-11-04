from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import masked_ttt.pipeline as pipeline
from experiments.mixed_on_policy import rollouts
from masked_ttt import Request
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_materialization_equivalence import gpt_gqa_engine
from test_public_linked_selection import _source
from test_standalone import make_engine


def fake_engine():
    def lm_head(_hidden):return torch.zeros((1,16),dtype=torch.float32)
    return SimpleNamespace(device=torch.device("cpu"),eos_ids=set(),budget=SimpleNamespace(check=lambda:None),
                           model=SimpleNamespace(lm_head=lm_head))


def test_sampling_uses_a_private_reproducible_generator_without_global_rng(monkeypatch):
    engine=fake_engine()
    reader=SimpleNamespace(memory=SimpleNamespace(document_length=9),reader=object(),
                           compiled_branch=lambda _cache:object())
    monkeypatch.setattr(rollouts,"private_forward",lambda *_args,**_kwargs:
                        SimpleNamespace(last_hidden_state=torch.zeros((1,1,4))))
    before=torch.get_rng_state().clone()
    generate=lambda:rollouts.sampled_rollout(engine,reader,object(),torch.tensor([1,2]),6,
                                             torch.Generator(device="cpu").manual_seed(41))
    first,second=generate(),generate()
    assert torch.equal(first,second) and len(first)==6 and bool((first!=0).any())
    assert torch.equal(before,torch.get_rng_state())


def test_mixed_context_alternates_and_restores_original_on_error(monkeypatch):
    original=lambda *_args:torch.tensor([42])
    monkeypatch.setattr(pipeline,"student_rollout",original)
    sampled=[]
    def sample(_engine,_reader,_cache,_suffix,_limit,generator):
        sampled.append(generator.initial_seed());return torch.tensor([77])
    monkeypatch.setattr(rollouts,"sampled_rollout",sample)
    engine=fake_engine();args=(engine,None,None,torch.tensor([1]),1)
    with pytest.raises(RuntimeError,match="intentional"):
        with rollouts.mixed_rollouts(seed=42) as stats:
            assert pipeline.student_rollout(*args).tolist()==[42]
            assert pipeline.student_rollout(*args).tolist()==[77]
            assert pipeline.student_rollout(*args).tolist()==[42]
            assert stats["calls"]==3 and stats["sampled_calls"]==1 and stats["greedy_calls"]==2
            raise RuntimeError("intentional")
    assert pipeline.student_rollout is original and sampled==[42]
    with pytest.raises(ValueError,match="nonnegative"):
        with rollouts.mixed_rollouts(seed=-1):pass


@pytest.mark.parametrize("factory",[make_engine,gpt_gqa_engine])
def test_mixed_pair_keeps_teacher_selection_and_physical_cartridge(tmp_path,monkeypatch,factory):
    document,question,_,_=_source(tmp_path)
    engine=factory()
    config=replace(engine.config,keep_fraction=.1,steps=2,max_answer_steps=2,self_question_count=0,
        guard_enabled=False,selected_transfer_weight=0.,value_basis_source="document",
        release_full_before_cartridge=True,release_selected_source_before_decode=True)
    engine.config=config
    selector=LinkedFrozenSelector(tmp_path);selector.bind_case("case")
    request=Request(document=document,question=question,max_new_tokens=3)
    teacher_prefixes=[];original_score=pipeline.score_student_prefix
    def observe_teacher(engine_arg,reader,teacher,tokens):
        probabilities=original_score(engine_arg,reader,teacher,tokens)
        assert not probabilities.requires_grad
        teacher_prefixes.append(tokens.tolist())
        return probabilities
    monkeypatch.setattr(pipeline,"score_student_prefix",observe_teacher)
    before=torch.get_rng_state().clone()
    first=rollouts.run_mixed_pair(engine,selector,request,config)
    second=rollouts.run_mixed_pair(engine,selector,request,config)
    assert torch.equal(before,torch.get_rng_state())
    assert first["changed_settings"]==["student_prefix_sampling"]
    assert first["sampling"]["calls"]==2 and first["sampling"]["sampled_calls"]==1
    assert first["sampling"]["first_update"]=="greedy" and first["sampling"]["temperature"]==1.
    assert first["same_practice_checks"] and not first["reference_answers_used"]
    expected_prefixes=[step["student_rollout_tokens"] for pair in (first,second)
                       for side in ("left","right") for step in pair[side]["training"]["steps"]]
    assert teacher_prefixes==expected_prefixes
    for side in ("left","right"):
        result=first[side];memory=result["standalone_cartridge"]["memory"]
        assert result["training"]["objective"]=="on-policy KL(teacher || student)"
        assert result["training"]["selected_transfer_weight"]==0
        assert memory["kv_only_materialized"] and memory["additional_source_kv_bytes_before_decode"]==0
        assert memory["additional_training_adapter_bytes_before_decode"]==0
        assert result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"]==0
        assert result["training"]["steps"]==second[side]["training"]["steps"]
    initial=lambda r:r["training"]["steps"][0]
    assert initial(first["left"])["student_rollout_tokens"]==initial(first["right"])["student_rollout_tokens"]
    for field in ("teacher_KL_before","gain_gradient_norm","value_gradient_norm"):
        assert initial(first["left"])[field]==pytest.approx(initial(first["right"])[field],abs=1e-7)


def test_sampling_rejects_nonfinite_probabilities(monkeypatch):
    engine=fake_engine();engine.model.lm_head=lambda _hidden:torch.full((1,4),float("nan"))
    reader=SimpleNamespace(memory=SimpleNamespace(document_length=9),reader=object(),
                           compiled_branch=lambda _cache:object())
    monkeypatch.setattr(rollouts,"private_forward",lambda *_args,**_kwargs:
                        SimpleNamespace(last_hidden_state=torch.zeros((1,1,4))))
    with pytest.raises(FloatingPointError,match="Nonfinite"):
        rollouts.sampled_rollout(engine,reader,None,torch.tensor([1]),1,torch.Generator().manual_seed(1))
