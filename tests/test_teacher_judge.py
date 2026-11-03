import json

from masked_ttt.teacher_judge import assess,verified_focused_quotes


SOURCE='X originally had 300. X was finally approved at 320. Y was approved at 250.'


def data(**changes):
    result=dict(reason='A later update states the final amount.',
                evidence=['X was finally approved at 320.'],
                candidate_has_answer=True,candidate_contradicted=True)
    result.update(changes)
    return json.dumps(result)


def test_grounded_contradiction_rejects_before_training():
    result=assess(data(),SOURCE,policy='grounded_veto')
    assert result['parse_valid'] and result['quotes_verified'] and result['contradiction']
    assert not result['apply_update']


def test_uncertain_or_ungrounded_contradiction_does_not_veto():
    for text in (data(evidence=['invented quote']),data(evidence=[]),'not json'):
        result=assess(text,SOURCE,policy='grounded_veto')
        assert result['apply_update'] and not result['contradiction']
    assert not assess('not json',SOURCE,policy='grounded_veto',candidate_complete=False)['apply_update']
    assert not assess(data(candidate_has_answer=False),SOURCE,policy='grounded_veto')['apply_update']


def test_supported_policy_requires_both_verdict_and_source_quote():
    good=json.dumps(dict(reason='The final amount is stated.',
                         evidence=['X was finally approved at 320.'],candidate_supported=True))
    missing=json.dumps(dict(reason='I infer it.',evidence=[],candidate_supported=True))
    false=json.dumps(dict(reason='Conflicting value.',evidence=['X was finally approved at 320.'],
                          candidate_supported=False))
    assert assess(good,SOURCE,policy='support_required')['apply_update']
    assert not assess(missing,SOURCE,policy='support_required')['apply_update']
    assert not assess(false,SOURCE,policy='support_required')['apply_update']


def test_only_verified_selected_quotes_can_enter_full_judge():
    selected='X was finally approved at 320.'
    judgment=dict(parse_valid=True,evidence=[selected,'Y was approved at 250.','invented'])
    assert verified_focused_quotes(judgment,selected,SOURCE)==[selected]
    assert verified_focused_quotes(dict(judgment,parse_valid=False),selected,SOURCE)==[]
