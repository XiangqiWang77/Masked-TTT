import json

import pytest

from masked_ttt.frozen_scores import FrozenJSONSelector,load_scores


def test_cached_selection_is_bound_to_the_case_and_question(tmp_path):
    scores=tmp_path/'scores';scores.mkdir()
    (tmp_path/'case_to_score_file.json').write_text(json.dumps({'case1':'cached.json'}))
    source='First sentence. Second sentence.';question='Which sentence?'
    data=dict(schema='masked-ttt-source-scores-v1',complete=True,case_key='case1',question=question,
              reference_answers_used=False,source_spans=[[0,15],[15,len(source)]],
              records=[dict(chunk=0,utility_output_valid=True,decision_probabilities=dict(yes=.1,no=.9)),
                       dict(chunk=1,utility_output_valid=True,decision_probabilities=dict(yes=.9,no=.1))])
    (scores/'cached.json').write_text(json.dumps(data))
    selector=FrozenJSONSelector(tmp_path)
    selector.bind_case('case1')
    loaded=load_scores(tmp_path,source,question,selector.case_key,selector.by_case)
    assert loaded['records'][1]['decision_probabilities']['yes']==.9
    with pytest.raises(ValueError):
        load_scores(tmp_path,source,'Different question',selector.case_key,selector.by_case)
    with pytest.raises(KeyError):
        selector.bind_case('missing-case')
