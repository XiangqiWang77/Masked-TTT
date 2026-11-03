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
    assert effective==saved
    assert (config.keep_fraction,config.steps,config.value_rank)==(.1,2,8)
    assert (config.max_context,config.context_reserve_tokens)==(65536,4096)
    assert (config.student_rollout_tokens,config.self_question_count)==(8,2)
    assert (config.gain_learning_rate,config.value_learning_rate)==(.01,.0001)
    assert config.guard_enabled and config.standalone_cartridge_enabled
    assert load_method_config(path,steps=3).steps==3
    assert load_method_config(path,selected_transfer_weight=.25).selected_transfer_weight==.25


def test_config_file_rejects_invalid_json_shape_and_unknown_keys(tmp_path):
    path=tmp_path/'invalid.json';path.write_text('[]')
    with pytest.raises(ValueError):load_method_config(path)
    path.write_text('{"unknown_field":1}')
    with pytest.raises(TypeError):load_method_config(path)


def test_cached_selection_requires_an_explicit_case_id(tmp_path,monkeypatch,capsys):
    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(document='A short document.',question='What is it?')))
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model','unused','--request',str(request),
                                    '--score-json-dir',str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    assert '--case-id is required' in capsys.readouterr().err
