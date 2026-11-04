from dataclasses import replace

import pytest
import torch

from masked_ttt import TTTConfig
from masked_ttt.runtime import private_forward
from masked_ttt.standalone import answer_standalone,build_standalone
from test_selected_transfer import gpt_engine,setup_views
from test_standalone import make_engine


@torch.no_grad()
def manual_decode(engine,layout,selected,teacher,trained,limit,use_teacher_requirements):
    reader,branch,suffix,length,_=build_standalone(engine,layout,selected,teacher,trained)
    if use_teacher_requirements:suffix=teacher.question_tokens
    output=private_forward(reader.reader,suffix,length,branch)
    tokens=[]
    while len(tokens)<limit:
        token=int(engine.model.lm_head(output.last_hidden_state[0,-1:])[0].argmax())
        tokens.append(token)
        if token in engine.eos_ids or len(tokens)==limit:break
        output=private_forward(reader.reader,torch.tensor([token],device=engine.device),
            length+len(suffix)+len(tokens)-1,branch)
    return tokens


@pytest.mark.parametrize('kind',['qwen3','gpt_oss'])
def test_teacher_requirements_change_only_standalone_question_suffix(kind):
    engine=make_engine() if kind=='qwen3' else gpt_engine()
    _,layout,selected,_,teacher,trained,_=setup_views(engine)
    with torch.no_grad():
        trained.gain[...,selected[0]]=0.1
        trained.value_coefficients.normal_(0,.01)
    trained.value_enabled=True;trained.native_mode=False
    _,_,original_suffix,_,_=build_standalone(engine,layout,selected,teacher,trained)
    assert not torch.equal(original_suffix,teacher.question_tokens)
    old_values=[(layer.values._version,layer.values.clone()) for layer in teacher.document_cache.layers]
    for mode in ('original','teacher_requirements'):
        engine.config=replace(engine.config,standalone_answer_format=mode)
        actual=answer_standalone(engine,layout,selected,teacher,trained,3)
        expected=manual_decode(engine,layout,selected,teacher,trained,3,mode=='teacher_requirements')
        assert actual['generated_token_ids']==expected
        assert actual['answer_format']==mode and not actual['teacher_answer_in_final_prompt']
    for layer,(version,values) in zip(teacher.document_cache.layers,old_values):
        assert layer.values._version==version
        torch.testing.assert_close(layer.values,values,atol=0,rtol=0)


def test_invalid_answer_format_is_rejected():
    with pytest.raises(ValueError):TTTConfig(standalone_answer_format='teacher_answer')
