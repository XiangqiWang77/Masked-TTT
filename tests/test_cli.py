import json
import sys

import pytest

from masked_ttt.cli import main
from masked_ttt.config import TTTConfig


def test_fast_profile_defaults_match_the_submitted_configuration():
    config=TTTConfig()
    assert (config.steps,config.max_answer_steps,config.teacher_max_new_tokens)==(2,8,32)
    assert config.teacher_target_scope=='choice_or_all'
    assert config.answer_KL_stop==0.5


def test_cached_selection_requires_an_explicit_case_id(tmp_path,monkeypatch,capsys):
    request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(document='A short document.',question='What is it?')))
    monkeypatch.setattr(sys,'argv',['masked-ttt','--model','unused','--request',str(request),
                                    '--score-json-dir',str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    assert '--case-id is required' in capsys.readouterr().err
