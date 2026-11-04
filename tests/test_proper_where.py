from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import masked_ttt as runtime
import experiments.anchored_short_verifier.paired_controls as controls
from experiments.babilong_linked_selection.proper_where_selector import ProperWhereSelector
from experiments.babilong_linked_selection.question_overlap_selector import question_overlap
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.proper_where import report
from experiments.proper_where.monitor import development_decision
from experiments.proper_where.prepare import DEV,HOLDOUT,batch_script
from experiments.proper_where.runner import proper_where_runner
from masked_ttt import MaskedTTT,Request,TTTConfig
from test_public_linked_selection import _source
from test_standalone import make_engine


def test_runner_restores_hooks_and_freezes_original_olmo_settings():
    implementation=SimpleNamespace(ROOT="old",validate_manifest=lambda x:x,
                                   validate=lambda *_:None,replace=replace)
    old=(implementation.ROOT,implementation.validate_manifest,implementation.validate,
         implementation.replace,runtime.FrozenJSONSelector,controls.original_question_controls)
    manifest=dict(mode="development",models=["qwen"],selection_policy="decision_proper_where_v1",
        question_overlap_policy="proper_where",entity_reserve_fraction=.25,gain_head_mode="kv",steps=2,
        fixed_value_learning_rate=.0001,max_answer_steps=8,student_rollout_tokens=8,
        holdout_evaluated_at_selection=False)
    with pytest.raises(RuntimeError,match="intentional"):
        with proper_where_runner(implementation,runtime,controls,root="candidate",selector=ProperWhereSelector,
                                 physical_controls=kv_only_original_question_controls):
            assert implementation.validate_manifest(manifest)["selection_policy"]=="cached_decision_score10"
            config=implementation.replace(TTTConfig(),max_context=8192)
            assert config.steps==2 and config.max_answer_steps==8 and config.value_learning_rate==.0001
            assert config.gain_head_mode=="kv" and not config.align_student_teacher_prompt
            with pytest.raises(ValueError):
                implementation.validate_manifest(dict(manifest,fixed_value_learning_rate=.001))
            raise RuntimeError("intentional")
    assert (implementation.ROOT,implementation.validate_manifest,implementation.validate,
            implementation.replace,runtime.FrozenJSONSelector,controls.original_question_controls)==old


def test_end_to_end_cached_selection_original_sd_physical_decode_and_report(tmp_path,monkeypatch):
    document,_,spans,records=_source(tmp_path)
    question="Where is Daniel?"
    score_path=tmp_path/"scores/case.json";score_data=json.loads(score_path.read_text())
    score_data["question"]=question;score_path.write_text(json.dumps(score_data))
    engine=make_engine();engine.eos_ids={-1}
    implementation=SimpleNamespace(ROOT="old",validate_manifest=lambda x:x,
                                   validate=lambda *_:None,replace=replace)
    with proper_where_runner(implementation,runtime,controls,root="candidate",selector=ProperWhereSelector,
                             physical_controls=kv_only_original_question_controls):
        engine.config=implementation.replace(engine.config,max_context=engine.config.max_context,
            keep_fraction=.1,student_rollout_tokens=8,self_question_count=0,guard_enabled=False,
            selected_transfer_weight=0.,value_basis_source="document",release_full_before_cartridge=True,
            release_selected_source_before_decode=True,teacher_response_instruction="Return only the final answer.")
        selector=runtime.FrozenJSONSelector(tmp_path);selector.bind_case("case")
        with controls.original_question_controls():
            result=MaskedTTT(engine,selector).answer(Request(document=document,question=question,max_new_tokens=8))
        implementation.validate(result,{"key":"case"})
    assert result["selection"]["selected_chunks"]==question_overlap(document,question,spans,records,.1,mode="proper_where")
    assert result["training"]["objective"]=="on-policy KL(teacher || student)"
    memory=result["standalone_cartridge"]["memory"]
    assert memory["kv_only_materialized"] and memory["cartridge_adapter_bytes"]==0
    assert memory["additional_source_kv_bytes_before_decode"]==0
    assert memory["additional_training_adapter_bytes_before_decode"]==0
    result.update(key="case",model="qwen-test",model_short="qwen",task="qa1",length=8192,keep_fraction=.1)
    result["self_judge"]["native_prediction"]="blue"
    old=deepcopy(result);old["selection"]["selection_policy"]="decision_entity_link_v1"
    before=old["standalone_cartridge"]["original_question_controls"]
    old["standalone_cartridge"]["paired_prompt_control"]=dict(zero_original=before["zero"],
        trained_original=before["adapted"],same_selected_kv=True,same_generation_limit=True,
        reference_answers_used=False,teacher_answer_in_student_prompt=False)
    report.validate_old(old,result,"case")
    invalid=deepcopy(old);invalid["selection"]["original_prompt_tokens"]+=1
    with pytest.raises(ValueError):report.validate_old(invalid,result,"case")
    for folder,value in ((tmp_path/"candidate",result),(tmp_path/"baseline",old)):
        path=folder/"cases/qwen/case/result.json";path.parent.mkdir(parents=True);path.write_text(json.dumps(value))
    dev=["case"]+[f"missing-{i}" for i in range(199)];held=[f"held-{i}" for i in range(800)]
    manifest=dict(mode="development",models=["qwen"],case_count=200,case_ids=dev,
        development_case_ids=dev,holdout_case_ids=held,selection_policy="decision_proper_where_v1",
        question_overlap_policy="proper_where",entity_reserve_fraction=.25,keep_fraction=.1,
        expected_development_changed_selections=7,
        student_rollout_tokens=8,gain_head_mode="kv",fixed_value_learning_rate=.0001,max_answer_steps=8,
        selected_transfer_weight=0.,reference_answers_used_in_inference=False,online_api_calls=0,
        holdout_evaluated_at_selection=False)
    monkeypatch.setattr(report,"validate_partition",lambda *_:None)
    out=report.collect(manifest,tmp_path/"candidate",tmp_path/"baseline",
        {"case":dict(task="qa1",answers=["blue"])},lambda text,*_:(int("blue" in text),))
    assert not out["errors"] and not out["invalid"] and not out["unmatched"]
    assert out["overall"]["n"]==1 and not development_decision(out)["eligible"]


def test_promotion_requires_complete_paired_improvement_and_one_gpu():
    value=dict(mode="development",expected=200,metric="answer_focused_lexical_v2",errors=[],invalid=[],unmatched=[],
        expected_development_changed_selections=20,
        overall=dict(n=200,accuracy=dict(whole=50.,zero=52.,teacher=54.,old_cartridge=53.,cartridge=56.),
                     changed_selection=20))
    assert development_decision(value)["eligible"]
    value["overall"]["accuracy"]["teacher"]=56.
    assert not development_decision(value)["eligible"]
    for name in (DEV,HOLDOUT):
        script=batch_script(Path("/tmp")/name)
        assert "#SBATCH --gpus-per-task=1" in script and "proper_where/execute.py" in script
        assert "gpu:a40" in script and "gpu:l40s" in script
