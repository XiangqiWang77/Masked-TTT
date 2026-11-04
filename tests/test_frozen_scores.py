import json

import pytest

from masked_ttt import Request
from masked_ttt.frozen_scores import FrozenJSONSelector,IndexedSource,load_scores


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


def test_exact_source_verification_rejects_same_question_with_wrong_document(tmp_path, tiny_backbone):
    scores=tmp_path/'scores';scores.mkdir()
    (tmp_path/'case_to_score_file.json').write_text(json.dumps({'case1':'cached.json'}))
    document='First sentence. Second sentence.';question='Which sentence?'
    data=dict(schema='masked-ttt-source-scores-v1',complete=True,case_key='case1',question=question,
        reference_answers_used=False,source_spans=[[0,15],[15,len(document)]],
        canonical_tokenizer='synthetic',scoring_seconds=0.,estimated_cost_usd=0.,
        records=[dict(chunk=0,utility_output_valid=True,decision_probabilities=dict(yes=.1)),
                 dict(chunk=1,utility_output_valid=True,decision_probabilities=dict(yes=.9))])
    (scores/'cached.json').write_text(json.dumps(data))
    record=(json.dumps(dict(key='case1',document=document,query=question))+'\n').encode()
    source=tmp_path/'source.jsonl';source.write_bytes(b'\n'+record)
    index=tmp_path/'source_index.json';index.write_text(json.dumps([
        dict(key='case1',offset=1,bytes=len(record))]))
    selector=FrozenJSONSelector(tmp_path,source_index=index,source_cases=source)
    selector.bind_case('case1')
    prepared=selector.prepare(tiny_backbone,Request(document=document,question=question,max_new_tokens=3))
    assert prepared.selected==[1] and prepared.metadata['source_document_matches_index']
    changed='Other sentence. Second sentence.'
    assert len(changed)==len(document)
    with pytest.raises(ValueError,match='original document'):
        selector.prepare(tiny_backbone,Request(document=changed,question=question,max_new_tokens=3))
    with pytest.raises(ValueError,match='supplied together'):
        FrozenJSONSelector(tmp_path,source_index=index)


def test_indexed_source_does_not_use_evaluation_answers(tmp_path):
    record=(json.dumps(dict(key='case1',document='A',query='Q',answers=['A']))+'\n').encode()
    source=tmp_path/'source.jsonl';source.write_bytes(record)
    index=tmp_path/'index.json';index.write_text(json.dumps([dict(key='case1',offset=0,bytes=len(record))]))
    checker=IndexedSource(index,source)
    with pytest.raises(ValueError,match='must not contain evaluation answers'):
        checker.verify('case1','A','Q')
    with pytest.raises(KeyError):
        checker.verify('missing','A','Q')
    index.write_text(json.dumps([dict(key='case1',offset=0,bytes=len(record)+1)]))
    with pytest.raises(ValueError,match='incomplete'):
        IndexedSource(index,source).verify('case1','A','Q')
