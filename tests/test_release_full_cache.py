from dataclasses import replace
import weakref

import pytest

from masked_ttt import MaskedTTT,Request,TTTConfig
from masked_ttt.memory_accounting import cache_tensors
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


@pytest.mark.parametrize("architecture",["qwen3","gpt_oss"])
def test_release_precedes_cartridge_decode_and_preserves_answers(architecture,monkeypatch):
    import masked_ttt.pipeline as pipeline
    engine=make_engine() if architecture=="qwen3" else gpt_engine()
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        standalone_cartridge_enabled=True,selected_transfer_weight=0.,release_full_before_cartridge=False)
    document="Old note says red. Current note says blue. "*6
    request=Request(document=document,question="What is in the current note?",max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend="unit_test"))
    method=MaskedTTT(engine,Selector())
    normal=method.answer(request)
    engine.config=replace(engine.config,release_full_before_cartridge=True)
    original_prefill=pipeline.prefill_document;original_standalone=pipeline.answer_standalone
    refs=[];observed=[]
    def tracked_prefill(*args,**kwargs):
        cache=original_prefill(*args,**kwargs)
        refs.extend(weakref.ref(t) for t in cache_tensors(cache))
        return cache
    def checked_standalone(*args,**kwargs):
        assert refs and all(ref() is None for ref in refs)
        observed.append(True)
        return original_standalone(*args,**kwargs)
    monkeypatch.setattr(pipeline,"prefill_document",tracked_prefill)
    monkeypatch.setattr(pipeline,"answer_standalone",checked_standalone)
    released=method.answer(request)
    assert observed==[True]
    assert released["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"]==0
    assert released["standalone_cartridge"]["generated_token_ids"]==normal["standalone_cartridge"]["generated_token_ids"]
    assert released['prediction']==released['standalone_cartridge']['prediction']
    assert released['generated_token_ids']==released['standalone_cartridge']['generated_token_ids']
    assert released['output_source']=='standalone_cartridge'
    assert released['self_judge']['applies_to']=='full_context_diagnostic'
    assert released['timing']['ttft_seconds']>=released['timing']['full_context_ttft_seconds']
    assert released["teacher"]["tokens"]==normal["teacher"]["tokens"]
    assert released["training"]["steps"]==normal["training"]["steps"]
    assert released["backbone_unchanged"] and released["original_document_KV_unchanged"]


def test_release_requires_standalone_generation():
    with pytest.raises(ValueError):
        TTTConfig(standalone_cartridge_enabled=False,release_full_before_cartridge=True)


def test_primary_answer_is_from_cartridge_even_if_full_answer_differs(monkeypatch):
    import masked_ttt.pipeline as pipeline
    engine=make_engine()
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        standalone_cartridge_enabled=True,release_full_before_cartridge=True)
    request=Request(document='Old note says red. Current note says blue. '*6,
                    question='What is in the current note?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    sentinel='independent-cartridge-output'
    monkeypatch.setattr(pipeline,'answer_standalone',lambda *args:dict(
        prediction=sentinel,generated_token_ids=[3],first_token_seconds=0.,
        memory={},teacher_answer_in_final_prompt=False))
    result=MaskedTTT(engine,Selector()).answer(request)
    assert result['prediction']==sentinel and result['generated_token_ids']==[3]
    assert result['full_context']['prediction']!=sentinel
    assert result['output_source']=='standalone_cartridge'
    assert result['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']==0


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
@pytest.mark.parametrize('transfer_weight',[0.,.25])
def test_releasing_selected_source_preserves_output_and_counts_live_storage(architecture,transfer_weight):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.config=replace(engine.config,self_question_count=0,guard_enabled=False,
        selected_transfer_weight=transfer_weight,release_full_before_cartridge=True,
        release_selected_source_before_decode=False)
    request=Request(document='Old note says red. Current note says blue. '*6,
                    question='What is in the current note?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    method=MaskedTTT(engine,Selector())
    retained=method.answer(request)
    engine.config=replace(engine.config,release_selected_source_before_decode=True)
    released=method.answer(request)
    before=retained['standalone_cartridge']['memory']
    after=released['standalone_cartridge']['memory']
    assert not before['source_release_requested'] and after['source_release_requested']
    assert before['additional_source_kv_bytes_before_decode']>0
    assert after['additional_source_kv_bytes_before_decode']==0
    assert after['live_document_state_before_decode_bytes']>=after['standalone_cartridge_bytes']
    assert after['live_document_state_before_decode_bytes']<before['live_document_state_before_decode_bytes']
    assert released['generated_token_ids']==retained['generated_token_ids']
    assert released['training']['steps']==retained['training']['steps']
    assert released['backbone_unchanged'] and released['original_document_KV_unchanged']


@pytest.mark.parametrize('architecture',['qwen3','gpt_oss'])
def test_zero_update_reuses_only_the_required_selected_cache_storage(architecture):
    engine=make_engine() if architecture=='qwen3' else gpt_engine()
    engine.config=replace(engine.config,steps=0,self_question_count=0,guard_enabled=False)
    request=Request(document='Old note says red. Current note says blue. '*6,
                    question='What is in the current note?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    result=MaskedTTT(engine,Selector()).answer(request)
    usage=result['standalone_cartridge']['memory']
    assert usage['source_release_requested']
    assert usage['additional_source_kv_bytes_before_decode']==0
    assert usage['live_document_state_before_decode_bytes']>=usage['standalone_cartridge_bytes']
    assert result['prediction']==result['standalone_cartridge']['prediction']
