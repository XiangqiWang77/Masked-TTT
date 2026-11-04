import math
from dataclasses import replace

import pytest

from experiments.counterfactual_transfer.probe import _mentions_only,make_pair,run_pair
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def test_only_an_unselected_connection_changes():
    a,b=make_pair();doc_a,spans_a,_=a;doc_b,spans_b,_=b
    assert len(spans_a)==len(spans_b)==20
    assert [doc_a[s:e] for s,e in (spans_a[0],spans_a[-1])]==[
        doc_b[s:e] for s,e in (spans_b[0],spans_b[-1])]
    assert 'garden' in doc_a[spans_a[9][0]:spans_a[9][1]]
    assert 'office' in doc_b[spans_b[9][0]:spans_b[9][1]]


@pytest.mark.parametrize('model',['qwen','gpt'])
def test_counterfactual_probe_preserves_selected_input_and_full_release(model):
    engine=make_engine() if model=='qwen' else gpt_engine()
    engine.config=replace(engine.config,release_full_before_cartridge=True,
                          selected_transfer_weight=0.)
    old=engine.config
    result=run_pair(engine,steps=1,max_new_tokens=3,rollout_tokens=2)
    assert engine.config is old and result['selected_inputs_identical']
    assert result['whole_cache_released_before_cartridge']
    assert result['candidates_used_after_training_only']
    assert not result['teacher_answer_in_final_prompt']
    assert result['zero_cache_max_logprob_difference']<1e-4
    assert result['answers'][0]['zero']==result['answers'][1]['zero']
    assert result['answers'][0]['teacher']==result['answers'][1]['teacher']
    assert all(x['training_steps']==1 for x in result['answers'])
    assert math.isfinite(result['margin_difference_garden_minus_office_context'])
    assert all(math.isfinite(v) for v in result['state_difference_norms'].values())


def test_basic_answer_mention_check_rejects_negation_and_both_candidates():
    assert _mentions_only('In the garden.','garden','office')
    assert not _mentions_only('Not in the garden.','garden','office')
    assert not _mentions_only('Garden or office.','garden','office')


def test_failed_probe_restores_runtime_hooks_and_backbone_config(monkeypatch):
    import masked_ttt.pipeline as pipeline
    engine=make_engine();old_config=engine.config;old_answer=pipeline.answer_standalone
    def failed_prefill(*args,**kwargs):raise RuntimeError('intentional test failure')
    monkeypatch.setattr(pipeline,'prefill_document',failed_prefill)
    with pytest.raises(RuntimeError,match='intentional'):
        run_pair(engine,steps=1,max_new_tokens=2,rollout_tokens=2)
    assert pipeline.prefill_document is failed_prefill
    assert pipeline.answer_standalone is old_answer
    assert engine.config is old_config
