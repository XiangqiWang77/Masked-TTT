from dataclasses import replace
import json
import math

import pytest

from experiments.babilong_linked_selection.linked_frozen import LinkedFrozenSelector as ExperimentalSelector
from experiments.babilong_linked_selection.selector import choose as experimental_choose
from experiments.kv_only_cartridge.paired_control import physical_paired_prompt_control
from masked_ttt import MaskedTTT, Request
from masked_ttt.linked_selection import LinkedFrozenSelector, choose
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def _source(tmp_path):
    chunks=["Other."]*40
    chunks[1]="Daniel grabbed the milk."
    chunks[2]="Daniel moved to the garden."
    chunks[6]="Daniel dropped the milk."
    document="\n".join(chunks)
    question="Where is the milk?"
    spans=[];cursor=0
    for text in chunks:
        start=document.index(text,cursor);end=start+len(text)
        spans.append([start,end]);cursor=end
    records=[dict(chunk=i,utility_output_valid=True,
        decision_probabilities={"yes":{1:.9,2:.05,3:.8,6:.85}.get(i,.01)})
        for i in range(len(chunks))]
    (tmp_path/"scores").mkdir()
    (tmp_path/"scores"/"case.json").write_text(json.dumps(dict(
        schema="masked-ttt-source-scores-v1",complete=True,case_key="case",
        question=question,reference_answers_used=False,source_spans=spans,
        canonical_tokenizer="test",scoring_seconds=0.,estimated_cost_usd=0.,records=records)))
    (tmp_path/"case_to_score_file.json").write_text(json.dumps({"case":"case.json"}))
    return document,question,spans,records


@pytest.mark.parametrize("policy",["score","neighbor","entity"])
@pytest.mark.parametrize("fraction",[.05,.1,.3])
def test_public_selector_matches_frozen_development_policy(tmp_path,policy,fraction):
    document,question,spans,records=_source(tmp_path)
    actual=choose(document,question,spans,records,fraction,policy,.25)
    expected=experimental_choose(document,question,spans,records,fraction,policy,.25)
    assert actual==expected
    assert len(actual)==max(1,math.ceil(len(records)*fraction))
    assert actual==sorted(set(actual))


@pytest.mark.parametrize("architecture",["qwen3","gpt_oss"])
def test_installed_selector_runs_physical_cartridge_without_network(architecture,tmp_path):
    document,question,spans,records=_source(tmp_path)
    engine=make_engine() if architecture=="qwen3" else gpt_engine()
    engine.config=replace(engine.config,keep_fraction=.1,self_question_count=0,
        guard_enabled=False,selected_transfer_weight=0.,value_basis_source="document",
        release_full_before_cartridge=True,release_selected_source_before_decode=True)
    request=Request(document=document,question=question,max_new_tokens=3)
    public=LinkedFrozenSelector(tmp_path);public.bind_case("case")
    old=ExperimentalSelector(tmp_path);old.bind_case("case")
    prepared=public.prepare(engine,request);reference=old.prepare(engine,request)
    assert prepared.selected==reference.selected==[1,2,3,6]
    assert prepared.metadata["selection_policy"]=="decision_entity_link_v1"
    assert not prepared.metadata["reference_answers_used"] and not prepared.metadata["online_api_calls"]
    with physical_paired_prompt_control():
        result=MaskedTTT(engine,public).answer(request)
    memory=result["standalone_cartridge"]["memory"]
    assert result["selection"]["selected_chunks"]==prepared.selected
    assert result["training"]["objective"]=="on-policy KL(teacher || student)"
    assert result["training"]["selected_transfer_weight"]==0
    assert result["backbone_unchanged"] and result["original_document_KV_unchanged"]
    assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"]==0
    assert memory["additional_source_kv_bytes_before_decode"]==0
    assert memory["additional_training_adapter_bytes_before_decode"]==0
