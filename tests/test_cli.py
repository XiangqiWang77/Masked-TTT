import json
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

import masked_ttt.cli as cli
from masked_ttt.cli import load_method_config,main
from masked_ttt.config import TTTConfig


def test_fast_profile_defaults_match_the_submitted_configuration():
    config=TTTConfig()
    assert (config.steps,config.max_answer_steps,config.teacher_max_new_tokens)==(2,8,32)
    assert config.teacher_target_scope=='choice_or_all'
    assert config.answer_KL_stop==0.5


def test_recorded_qwen_v5_config_is_loaded_without_changing_defaults():
    path=Path(__file__).resolve().parents[1]/'examples/qwen_babilong_v5_config.json'
    saved=json.loads(path.read_text())
    config=load_method_config(path,keep_fraction=None,steps=None)
    assert asdict(config)==saved
    assert (config.keep_fraction,config.steps,config.student_rollout_tokens)==(.3,4,32)
    assert config.teacher_judge_policy=='grounded_veto' and config.teacher_veto_against_native
    assert config.practice_when_teacher_rejected and config.adaptive_open_ended
    assert load_method_config(path,keep_fraction=.2).keep_fraction==.2
    assert load_method_config().keep_fraction==.1


def test_config_file_must_be_a_valid_ttt_config(tmp_path):
    path=tmp_path/'invalid.json';path.write_text('[]')
    with pytest.raises(ValueError):load_method_config(path)
    path.write_text('{"unknown_field":1}')
    with pytest.raises(TypeError):load_method_config(path)


def test_cli_passes_recorded_config_to_backbone(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'examples/qwen_babilong_v5_config.json'
    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(document='A short document.',question='What is it?')))
    output=tmp_path/'output.json';seen={}
    class Selector:
        def __init__(self,directory):seen['scores']=directory
        def bind_case(self,key):seen['case']=key
    class Model:
        @staticmethod
        def load(model,config,**kwargs):seen['config']=config;return object()
    class Method:
        def __init__(self,backbone,selector):pass
        def answer(self,request,time_budget_seconds=None):return dict(prediction='done')
    monkeypatch.setattr(cli,'FrozenJSONSelector',Selector)
    monkeypatch.setattr(cli,'Backbone',Model)
    monkeypatch.setattr(cli,'MaskedTTT',Method)
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model','unused','--request',str(request),
        '--score-json-dir',str(tmp_path),'--case-id','sample','--config-json',str(path),
        '--output',str(output)])
    main()
    assert asdict(seen['config'])==json.loads(path.read_text())
    assert seen['case']=='sample' and json.loads(output.read_text())['prediction']=='done'


def test_cached_selection_requires_an_explicit_case_id(tmp_path,monkeypatch,capsys):
    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(document='A short document.',question='What is it?')))
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model','unused','--request',str(request),
                                    '--score-json-dir',str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    assert '--case-id is required' in capsys.readouterr().err
