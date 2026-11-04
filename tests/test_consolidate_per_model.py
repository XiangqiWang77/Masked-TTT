import copy

import pytest

from experiments.babilong_linked_selection.consolidate_per_model import consolidate, render


def fixture():
    def ids(start, count):
        return [f'babilong-{length}-qa{number%10+1}-{300000+number}'
                for number in range(start, start+count) for length in (8,16,32,64)]
    dev_keys, held_keys = ids(0,50), ids(50,200)
    common = dict(models=['qwen','gpt'], keep_fraction=.1, selection_policy='decision_entity_link_v1',
        entity_reserve_fraction=.25, basis_source='document', selected_transfer_weight=0.,
        reference_answers_used_in_inference=False, online_api_calls=0)
    development_manifest = dict(common, mode='development', case_count=200, case_ids=dev_keys,
                                holdout_case_ids=held_keys)
    holdout_manifest = dict(common, mode='holdout', case_count=800, case_ids=held_keys,
        holdout_evaluated_at_selection=False,
        qualitative_case_sources=[held_keys[0].split('-',2)[2],held_keys[4].split('-',2)[2]])
    def rows(keys):
        result=[]
        for index,key in enumerate(keys):
            fields=key.split('-',3)
            yes=int(index%2==0)
            result.append(dict(case=key,task=fields[2],length=int(fields[1])*1024,
                whole=0,baseline10=0,zero=0,teacher=0,cartridge=yes,
                zero_teacher_format=0,cartridge_teacher_format=yes,
                same_full_answer=True,baseline_source='full_1000'))
        return result
    def report(keys, *, development):
        return dict(updated=0.,metric='answer_focused_lexical_v2',development_only=development,
            expected_per_model=200 if development else 800,selection_policy='decision_entity_link_v1',
            selected_fraction=.1,baseline_fraction=.1,cached_scores=True,online_api_calls=0,
            evaluation_answers_used_only_posthoc=True,errors=[],invalid=[],
            models={'qwen':{'rows':rows(keys)},'gpt':{'rows':rows(keys if development else keys[:7])}})
    return development_manifest,holdout_manifest,report(dev_keys,development=True),report(held_keys,development=False)


def test_one_model_can_complete_while_other_model_is_pending():
    result=consolidate(*fixture())
    qwen,gpt=result['models']['qwen'],result['models']['gpt']
    assert qwen['state']=='complete' and qwen['combined']['n']==1000
    assert qwen['combined']['accuracy']['cartridge']==50.
    assert qwen['combined']['pairs']['sd_original_format']['new_correct']==500
    assert all(qwen['groups']['task:qa'+str(i)]['n']==100 for i in range(1,11))
    assert all(qwen['groups']['length:'+str(n)]['n']==250 for n in (8192,16384,32768,65536))
    assert gpt['state']=='in_progress' and gpt['holdout_count']==7 and 'combined' not in gpt
    assert 'gpt 留出集进行中' in render(result)


def test_incomplete_or_mismatched_cases_never_become_a_full_result():
    data=list(fixture());data[3]['models']['qwen']['rows'].pop()
    assert consolidate(*data)['models']['qwen']['state']=='in_progress'
    data=list(fixture());data[3]['models']['qwen']['rows'][1]['case']=data[3]['models']['qwen']['rows'][0]['case']
    with pytest.raises(ValueError):consolidate(*data)
    data=list(fixture());data[2]['models']['qwen']['rows'][0]['task']='qa9'
    with pytest.raises(ValueError):consolidate(*data)
    data=list(fixture());data[3]['models']['qwen']['rows'][0]['baseline_source']='other'
    with pytest.raises(ValueError):consolidate(*data)
    data=list(fixture());data[1]['keep_fraction']=.2
    with pytest.raises(ValueError):consolidate(*data)


def test_split_source_overlap_is_rejected():
    dev,held,report_a,report_b=copy.deepcopy(fixture())
    held['case_ids'][0]=dev['case_ids'][0]
    dev['holdout_case_ids'][0]=dev['case_ids'][0]
    with pytest.raises(ValueError):consolidate(dev,held,report_a,report_b)
