"""Freeze the 800-case selection-budget holdout after matched development."""
import math
import time


def source_id(case):
    parts=case.split('-',2)
    if len(parts)!=3 or parts[0]!='babilong' or parts[1] not in ('8','16','32','64'):
        raise ValueError('Unexpected BABILONG case identifier')
    return parts[2]


def build_manifest(development,latest):
    dev=development['case_ids'];holdout=development['holdout_case_ids']
    dev_sources={source_id(x) for x in dev};holdout_sources={source_id(x) for x in holdout}
    if (len(dev)!=200 or len(set(dev))!=200 or len(holdout)!=800 or len(set(holdout))!=800 or
        set(dev)&set(holdout) or len(dev_sources)!=50 or len(holdout_sources)!=200 or
        dev_sources&holdout_sources or development['keep_fraction']!=.1 or
        development.get('selection_policy')!='decision_entity_link_v1' or
        development.get('entity_reserve_fraction')!=.25 or development['selected_transfer_weight']!=0):
        raise ValueError('The reserved source-disjoint split changed')
    exposed=development.get('qualitative_case_sources',[])
    if (set(exposed)!={'qa1-300023','qa2-300037'} or len(exposed)!=2 or
        any(sum(source_id(key)==item for key in holdout)!=4 for item in exposed)):
        raise ValueError('Previously examined qualitative examples must be marked')
    if (latest['metric']!='answer_focused_lexical_v2' or not latest['development_only'] or
        latest['expected_per_model']!=200 or latest['selected_fraction']!=.1 or
        latest.get('selection_policy')!='decision_entity_link_v1' or
        latest['errors'] or latest['invalid']):
        raise ValueError('Require a clean linked-selection development report')
    selected=[];decisions={}
    for model in ('qwen','gpt'):
        item=latest['models'][model];s=item['overall'];matched=item['matched_10']
        if s['n']!=200 or matched['n']!=200:
            raise RuntimeError('Waiting for all 200 paired development answers: '+model)
        rows=item.get('rows',[])
        if (len(rows)!=200 or {row['case'] for row in rows}!=set(dev) or
            any(row.get('baseline10') is None or row.get('baseline_source') not in
                ('full_1000','value_lr_development') for row in rows)):
            raise ValueError('Development baseline provenance is incomplete')
        a=s['accuracy'];values=[a[k] for k in ('whole','zero','teacher','cartridge')]
        values += [matched['baseline10_accuracy'],matched['candidate_accuracy']]
        if any(type(x) not in (float,int) or not math.isfinite(x) for x in values):
            raise ValueError('Incomplete development accuracy')
        for key in ('whole','zero','teacher','cartridge','baseline10'):
            if any(row[key] not in (0,1) for row in rows):
                raise ValueError('Invalid per-case score')
            actual=100*sum(row[key] for row in rows)/len(rows)
            stated=(matched['baseline10_accuracy'] if key=='baseline10' else a[key])
            if not math.isclose(actual,stated,abs_tol=1e-9):
                raise ValueError('Development summary differs from paired rows')
        # Both values score the same 200 candidate predictions.
        if not math.isclose(a['cartridge'],matched['candidate_accuracy'],abs_tol=1e-9):
            raise ValueError('The 10% comparison used a different candidate cohort')
        beats_controls=a['cartridge']>max(a['whole'],a['zero'],a['teacher'])
        beats_ten=matched['candidate_accuracy']>matched['baseline10_accuracy']
        # Preserve the original, stricter OLMo-target decision. Separately
        # validate the selected-evidence improvement when it beats the paired
        # score-only cartridge without lowering its own same-format zero arm.
        # This second promotion rule was introduced after reviewing the
        # development results and must not be described as preregistered.
        beats_own_zero=a['cartridge']>=a['zero']
        strict_eligible=beats_controls and beats_ten
        selection_validation_eligible=beats_ten and beats_own_zero
        eligible=strict_eligible or selection_validation_eligible
        if eligible:selected.append(model)
        decisions[model]=dict(eligible=eligible,beats_own_full_zero_teacher=beats_controls,
            strict_olmo_target_eligible=strict_eligible,
            selection_validation_eligible=selection_validation_eligible,
            selection_rule_added_after_development_review=True,
            beats_own_same_format_zero=beats_own_zero,
            beats_10_percent_cartridge=beats_ten,development_accuracy=a,
            versus_teacher=s['versus_teacher'],versus_zero=s['versus_zero'],
            versus_10_percent=matched,
            baseline_sources={row['case']:row['baseline_source'] for row in rows})
    return dict(scope='800 source-disjoint BABILONG holdout; no retuning',
        case_count=800,case_ids=sorted(holdout),models=selected,
        model_protocols=development['model_protocols'],
        random_order_seed=development['random_order_seed'],mode='holdout',
        keep_fraction=.1,selection_policy='decision_entity_link_v1',entity_reserve_fraction=.25,
        qualitative_case_sources=development['qualitative_case_sources'],
        basis_source='document',selected_transfer_weight=0.,
        reference_answers_used_in_inference=False,online_api_calls=0,max_gpu_hours=36,
        development_decisions=decisions,holdout_evaluated_at_selection=False,
        promotion_rule='Original strict rule: beat full, zero-update, selected Teacher and score-only '
                       'Cartridge. Additional selection-validation rule, added after development review: '
                       'beat score-only Cartridge and do not lower linked same-format zero-update accuracy.',
        selection_rule_added_after_development_review=True,
        selection_holdout_target='Linked Cartridge beats matched score-only 10% Cartridge and is not '
                                 'below linked same-format zero-update; tested separately from the OLMo target',
        holdout_target='At least +3 pp versus selected ICL Teacher and positive versus '
                       'full, selected zero-update and matched score-only 10% Cartridge; '
                       'exclude the two earlier qualitative examples from the target test',
        prepared=time.time())
