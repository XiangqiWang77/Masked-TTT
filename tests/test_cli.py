import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from masked_ttt.cli import load_method_config,main
from masked_ttt.config import TTTConfig


def test_fast_profile_defaults_match_the_submitted_configuration():
    config=TTTConfig()
    assert (config.steps,config.max_answer_steps,config.teacher_max_new_tokens)==(2,8,32)
    assert config.teacher_target_scope=='choice_or_all'
    assert config.answer_KL_stop==0.5


def test_recorded_olmo_profile_is_loaded_exactly_and_can_be_overridden():
    path=Path(__file__).resolve().parents[1]/'examples/olmo_babilong_icl_sd.json'
    saved=json.loads(path.read_text())
    config=load_method_config(path,keep_fraction=None,steps=None)
    effective=asdict(config)
    assert effective.pop('selected_transfer_weight')==0.0
    assert effective.pop('practice_boundary_fallback') is False
    assert effective.pop('release_full_before_cartridge') is True
    assert effective.pop('release_selected_source_before_decode') is True
    assert effective.pop('value_basis_source') == 'document'
    assert effective.pop('gain_head_mode') == 'kv'
    assert effective.pop('standalone_answer_format') == 'original'
    assert effective.pop('align_student_teacher_prompt') is False
    assert effective==saved
    assert (config.keep_fraction,config.steps,config.value_rank)==(.1,2,8)
    assert (config.max_context,config.context_reserve_tokens)==(65536,4096)
    assert (config.student_rollout_tokens,config.self_question_count)==(8,2)
    assert (config.gain_learning_rate,config.value_learning_rate)==(.01,.0001)
    assert config.guard_enabled and config.standalone_cartridge_enabled
    assert load_method_config(path,steps=3).steps==3
    assert load_method_config(path,selected_transfer_weight=.25).selected_transfer_weight==.25
    assert load_method_config(path,align_student_teacher_prompt=True).align_student_teacher_prompt
    assert load_method_config(path,value_learning_rate=.001,gain_learning_rate=.02).value_learning_rate==.001


def test_config_file_rejects_invalid_json_shape_and_unknown_keys(tmp_path):
    path=tmp_path/'invalid.json';path.write_text('[]')
    with pytest.raises(ValueError):load_method_config(path)
    path.write_text('{"unknown_field":1}')
    with pytest.raises(TypeError):load_method_config(path)


def test_two_backbone_profile_keeps_full_document_and_releases_both_source_caches():
    path=Path(__file__).resolve().parents[1]/'examples/query_cartridges_babilong.json'
    config=load_method_config(path)
    assert config.keep_fraction==.1 and config.max_context==131072
    assert config.gain_head_mode=='kv' and config.value_basis_source=='document'
    assert config.steps==2 and config.student_rollout_tokens==8 and config.selected_transfer_weight==0.
    assert config.release_full_before_cartridge and config.release_selected_source_before_decode


def test_cached_selection_requires_an_explicit_case_id(tmp_path,monkeypatch,capsys):
    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(document='A short document.',question='What is it?')))
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model','unused','--request',str(request),
                                    '--score-json-dir',str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    assert '--case-id is required' in capsys.readouterr().err


def test_cached_source_verification_requires_both_files(tmp_path,monkeypatch,capsys):
    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(document='A short document.',question='What is it?')))
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model','unused','--request',str(request),
        '--score-json-dir',str(tmp_path),'--case-id','case','--source-index','index.json'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2 and '--source-index and --source-jsonl' in capsys.readouterr().err


@pytest.mark.parametrize('architecture', ['qwen3', 'gpt_oss_gqa'])
def test_cli_returns_physically_independent_kv_only_cartridge(tmp_path,monkeypatch,architecture):
    from masked_ttt import Backbone
    from test_materialization_equivalence import gpt_gqa_engine
    from test_public_linked_selection import _source
    from test_standalone import make_engine

    document,question,_,_=_source(tmp_path)
    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(document=document,question=question,max_new_tokens=3)))
    source=tmp_path/'source.jsonl'
    record=(json.dumps(dict(key='case',document=document,query=question))+'\n').encode()
    source.write_bytes(record)
    index=tmp_path/'source_index.json'
    index.write_text(json.dumps([dict(key='case',offset=0,bytes=len(record))]))
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(chunk_tokens=32,max_context=2048,context_reserve_tokens=128,
        prefill_chunk_tokens=64,steps=2,max_answer_steps=2,value_rank=2,student_rollout_tokens=3,
        self_question_count=0,guard_enabled=False,selected_transfer_weight=0.,
        release_full_before_cartridge=True,release_selected_source_before_decode=True)))
    result_path=tmp_path/'result.json'
    engine=make_engine() if architecture=='qwen3' else gpt_gqa_engine()
    engine.eos_ids={-1}
    def load(_cls,_path,config,**_kwargs):
        engine.config=config
        engine.cfg=dict(max_context=config.max_context,chunk_tokens=config.chunk_tokens,
                        chunk_boundary_mode='legacy',prefill_chunk_tokens=config.prefill_chunk_tokens)
        return engine
    monkeypatch.setattr(Backbone,'load',classmethod(load))
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model','synthetic','--request',str(request),
        '--score-json-dir',str(tmp_path),'--case-id','case','--config-json',str(profile),
        '--source-index',str(index),'--source-jsonl',str(source),
        '--value-learning-rate','0.001','--gain-learning-rate','0.02','--output',str(result_path)])
    main()
    result=json.loads(result_path.read_text())
    assert result['output_source']=='standalone_cartridge'
    assert result['prediction']==result['standalone_cartridge']['prediction']
    assert result['training']['objective']=='on-policy KL(teacher || student)'
    assert result['training']['gain_learning_rate']==.02 and result['training']['value_learning_rate']==.001
    assert result['training']['selected_transfer_weight']==0.
    assert any(step['gain_gradient_norm']>0 for step in result['training']['steps'])
    assert any(step['value_gradient_norm']>0 for step in result['training']['steps'])
    assert result['selection']['backend']=='precomputed_source_JSON'
    assert result['selection']['source_document_matches_index']
    assert result['selection']['online_api_calls']==0 and not result['selection']['reference_answers_used']
    assert result['backbone_unchanged'] and result['original_document_KV_unchanged']
    assert result['memory']['full_cache_release_before_standalone']['full_cache_tensors_live_before_standalone']==0
    memory=result['standalone_cartridge']['memory']
    assert memory['kv_only_materialized'] and memory['cartridge_adapter_bytes']==0
    assert memory['additional_source_kv_bytes_before_decode']==0
    assert memory['additional_training_adapter_bytes_before_decode']==0


def test_cli_respects_explicit_retention_of_selected_source_cache(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import masked_ttt.cli as cli

    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(document='A short document.',question='What is it?')))
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(release_selected_source_before_decode=False)))
    result_path=tmp_path/'result.json'
    selector=SimpleNamespace(bind_case=lambda _case:None)
    monkeypatch.setattr(cli,'FrozenJSONSelector',lambda _path:selector)
    monkeypatch.setattr(cli.Backbone,'load',lambda *_args,**_kwargs:object())
    monkeypatch.setattr(cli,'MaskedTTT',lambda *_args:SimpleNamespace(
        answer=lambda *_a,**_kw:dict(prediction='ok')))
    monkeypatch.setattr(cli,'kv_only_cartridge',lambda:pytest.fail('source release was disabled'))
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model','synthetic','--request',str(request),
        '--score-json-dir',str(tmp_path),'--case-id','case','--config-json',str(profile),
        '--output',str(result_path)])
    main()
    assert json.loads(result_path.read_text())['prediction']=='ok'
