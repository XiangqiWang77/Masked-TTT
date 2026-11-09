"""Score configured SD arms on the same completed cases; references stay here only."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time

from experiments.run_syn_sd_factorial import read, write


ARMS = ('vanilla_syns_oracle_sd', 'vanilla_syns_selective_sd',
        'specific_syns_oracle_sd', 'specific_syns_selective_sd')


def official_evaluator():
    from experiments.run_babilong import REFERENCES
    path = Path('/home/da839/Stack_TTT/third_party/qa_suite/babilong_metrics.py')
    spec = importlib.util.spec_from_file_location('syn_sd_official_babilong', path)
    metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
    return read(REFERENCES), metrics, str(path)


def aggregate(rows, arm_names=ARMS):
    complete = [row for row in rows if all(arm in row['scores'] for arm in arm_names)]
    arms = {}
    for arm in arm_names:
        observed = [row['scores'][arm] for row in rows if arm in row['scores']]
        paired = [row['scores'][arm] for row in complete]
        arms[arm] = {'observed_n': len(observed), 'observed_correct': sum(observed),
            'complete_paired_n': len(paired), 'complete_paired_correct': sum(paired),
            'paired_accuracy_percent': 100*sum(paired)/len(paired) if paired else None}
    contrasts = {}
    comparisons=[];dataset_pairs={}
    for dataset in ('vanilla_syns','specific_syns'):
        baseline=next((dataset+'_'+sd for sd in ('vanilla_sd','oracle_sd')
                       if dataset+'_'+sd in arm_names),None)
        selective=dataset+'_selective_sd'
        if baseline and selective in arm_names:
            dataset_pairs[dataset]=(baseline,selective)
            comparisons.append(('selective_with_'+dataset,baseline,selective))
    if len(dataset_pairs)==2:
        for index,label in ((0,'vanilla_SD'),(1,'selective_SD')):
            comparisons.append(('query_specific_with_'+label,
                dataset_pairs['vanilla_syns'][index],dataset_pairs['specific_syns'][index]))
    for name, baseline, target in comparisons:
        if baseline not in arm_names or target not in arm_names: continue
        fixes = [row['case'] for row in complete if not row['scores'][baseline] and row['scores'][target]]
        harms = [row['case'] for row in complete if row['scores'][baseline] and not row['scores'][target]]
        contrasts[name] = {'n': len(complete), 'fixes': len(fixes), 'regressions': len(harms),
            'delta_pp': 100*(len(fixes)-len(harms))/len(complete) if complete else None,
            'fixed_cases': fixes, 'regressed_cases': harms}
    beyond = {}
    for arm in arm_names:
        pairs = [row for row in complete if 'full_context_teacher' in row['baseline_scores']]
        fixes = sum(not row['baseline_scores']['full_context_teacher'] and row['scores'][arm] for row in pairs)
        harms = sum(row['baseline_scores']['full_context_teacher'] and not row['scores'][arm] for row in pairs)
        beyond[arm] = {'n': len(pairs), 'teacher_wrong_student_correct': fixes,
            'teacher_correct_student_wrong': harms,
            'teacher_wrong_initial_wrong_student_correct': sum(
                not row['baseline_scores']['full_context_teacher']
                and row['baseline_scores'].get('initial_kv') == 0 and row['scores'][arm] for row in pairs),
            'delta_pp': 100*(fixes-harms)/len(pairs) if pairs else None}
    oracle={}
    for dataset,(baseline,selective) in dataset_pairs.items():
        paired=[row for row in rows if baseline in row['scores'] and selective in row['scores']
                and 'full_context_teacher' in row['baseline_scores']]
        select=[row for row in paired if not row['scores'][baseline] and row['scores'][selective]]
        regress=[row for row in paired if row['scores'][baseline] and not row['scores'][selective]]
        teacher_wrong=[row for row in paired if not row['baseline_scores']['full_context_teacher']]
        corrected=[row for row in teacher_wrong if row['scores'][selective]]
        new_corrections=[row for row in corrected if row['baseline_scores'].get('initial_kv')==0
                         and row['training'][selective]['updates']>0]
        oracle[dataset]={'paired_cases':len(paired),
            'vanilla_SD_correct':sum(row['scores'][baseline] for row in paired),
            'selective_SD_correct':sum(row['scores'][selective] for row in paired),
            'selective_fixes':len(select),'selective_regressions':len(regress),
            'selective_net_delta_pp':100*(len(select)-len(regress))/len(paired) if paired else None,
            'teacher_wrong_cases':len(teacher_wrong),
            'teacher_wrong_selective_correct':len(corrected),
            'teacher_wrong_vanilla_wrong_selective_correct':sum(not row['scores'][baseline] for row in corrected),
            'teacher_wrong_initial_wrong_updated_selective_correct':len(new_corrections),
            'teacher_wrong_initial_wrong_vanilla_wrong_updated_selective_correct':sum(
                not row['scores'][baseline] for row in new_corrections),
            'teacher_wrong_initial_correct_selective_correct':sum(
                row['baseline_scores'].get('initial_kv')==1 for row in corrected),
            'teacher_correct_selective_wrong':sum(row['baseline_scores']['full_context_teacher']
                and not row['scores'][selective] for row in paired),
            'selective_zero_update_cases':sum(row['training'][selective]['updates']==0 for row in paired),
            'fixed_case_ids':[row['case'] for row in select],
            'regressed_case_ids':[row['case'] for row in regress],
            'new_correction_case_ids':[row['case'] for row in new_corrections]}
    value = {'planned': len(rows), 'complete_case_count': len(complete), 'arms': arms,
             'contrasts': contrasts, 'versus_full_context_teacher': beyond,
             'oracle_trap_evidence':oracle}
    value['training_budget_audit'] = {
        'complete_cases':len(complete),
        'actual_updates_match_all_arms':sum(len({r['training'][a]['updates'] for a in arm_names})==1
                                          for r in complete),
        'zero_update_cases':{a:sum(r['training'][a]['updates']==0 for r in complete) for a in arm_names},
        'all_rejected_original_SD_fallback_cases':{
            a:sum(r['training'][a].get('all_rejected_original_SD_fallback',False) for r in complete)
            for a in arm_names}}
    if tuple(arm_names) == ARMS: value['four_way_complete'] = len(complete)
    return value


def build(config):
    root = Path(config['output']); references, metrics, metric_path = official_evaluator()
    arm_names = tuple(config.get('matrix', ARMS))
    partition_path = root/'evaluation_partitions.json'
    partitions = read(partition_path) if partition_path.is_file() else None
    from contextualized.data import read_indexed_case
    plan = read(config['source_plan'])
    index = {entry['key']: entry for entry in read(plan['source_index'])}
    rows, issues = [], []
    for item in config['cases']:
        key = item['key']; base = root/'cases'/key
        row = {'case': key, 'task': item['task'], 'length': item['length'],
               'analysis_split':item.get('analysis_split','unspecified'),
               'scores': {}, 'baseline_scores': {}, 'training': {}, 'predictions': {}}
        if partitions is not None:
            row.update(partitions['cases'][key])
        result = root/'results/cases'/(key+'.json')
        record = read(result) if result.is_file() else None
        reference = references[key]
        query = record['query'] if record else item.get('query')
        if query is None:
            query = read_indexed_case(plan, index, key)['query']
        def score(prediction):
            return int(any(metrics.compare_answers(answer, prediction, query,
                metrics.TASK_LABELS[item['task']]) for answer in reference['answers']))
        try:
            baseline = base/'baselines.json'
            if baseline.is_file():
                data = read(baseline)
                for name in ('full_context_teacher', 'no_context', 'initial_kv'):
                    if isinstance(data.get(name), str): row['baseline_scores'][name] = score(data[name])
            for arm in arm_names:
                path = base/'arms'/(arm+'.json')
                if not path.is_file(): continue
                value = read(path)
                if value.get('case') != key or value.get('status') != 'completed':
                    raise ValueError('Invalid saved arm result')
                row['scores'][arm] = score(value['prediction'])
                row['predictions'][arm] = value['prediction']
                row['training'][arm] = {'updates': value['training']['updates'],
                    'train_conversations': value['training']['train_conversations'],
                    'parameter_change_l2': value['native_kv_parameter_change_l2'],
                    'seconds': value['seconds'], 'no_accepted_signal': value['no_accepted_signal'],
                    'returned_checkpoint_step':value['training'].get('returned_checkpoint_step'),
                    'all_rejected_original_SD_fallback':value['training'].get('all_rejected_original_SD_fallback',False)}
        except Exception as error:
            issues.append({'case': key, 'error': str(error)})
        rows.append(row)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': config['protocol'], 'metric': 'official_BABILong_label_accuracy',
        'metric_implementation': metric_path, 'official_rules_modified': False,
        'planned': len(rows), 'matrix': list(arm_names), 'overall': aggregate(rows, arm_names),
        'by_length': {str(length): aggregate([row for row in rows if row['length'] == length], arm_names)
                      for length in (8192, 16384, 32768, 65536)},
        'by_task': {f'qa{n}': aggregate([row for row in rows if row['task'] == f'qa{n}'], arm_names) for n in range(1,11)},
        'by_analysis_split':{name:aggregate([row for row in rows if row['analysis_split']==name],arm_names)
                             for name in ('development','holdout','unspecified')},
        'issues': issues, 'failure_cases': [p.stem for p in (root/'results/failures').glob('*.json')],
        'historical_execution_failure_records':sum(1 for _ in (root/'results/infrastructure_history').glob('**/*.json')),
        'cases': rows, 'references_used_only_after_inference': True,
        'question_budget_policy': config['question_budget_policy'],
        'missing_predictions_are_not_scored_as_wrong': True,
        'accuracy_comparisons_use_same_completed_cases': True,
        'vanilla_synthetic_used_for_SD': 'vanilla_syns' in config.get('dataset_arms', ('vanilla_syns','specific_syns'))}
    report['complete'] = report['overall']['complete_case_count'] == len(rows) and not issues
    if partitions is not None:
        from experiments.sd_contribution_audit import summarize_partitions
        report['contribution_validation'] = summarize_partitions(rows, arm_names, partitions)
    write(root/'report/accuracy.json', report)
    def display(value): return 'pending' if value is None else f'{value:.2f}'
    a = report['overall']['arms']; n = report['overall']['complete_case_count']
    lines = [f'Official BABILong accuracy (%); matched completed cases: {n}/{len(rows)}.', '',
        '| SD arm | Accuracy (%) | Completed paired cases |', '|---|---:|---:|']
    for arm in arm_names:
        lines.append('| '+arm+' | '+display(a[arm]['paired_accuracy_percent'])+f' | {n} |')
    teacher_label=('the original per-question source-conditioned Teacher' if config.get('teacher_context_policy')=='question_source_evidence'
                   else 'the frozen full-document Teacher')
    lines += ['', 'Vanilla SD uses '+teacher_label+' without admission filtering. '
        'Selective SD keeps the same Teacher targets only when the original source-supported gate accepts them. '
        'Synthetic answers are never labels. Missing predictions do not count as errors.']
    if config.get('dataset_arms') == ['specific_syns']:
        lines += ['', 'Only query-specific synthetic questions are trained. Vanilla synthetic data is excluded from this SD experiment.']
    destination = root/'report/summary.md'; destination.write_text('\n'.join(lines)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--watch-seconds', type=float)
    a = p.parse_args(); config = read(a.config)
    while True:
        report = build(config)
        print(json.dumps({'completed_cases': report['overall']['complete_case_count'],
            'planned': report['planned'], 'failures': len(report['failure_cases']),
            'issues': len(report['issues'])}), flush=True)
        if a.watch_seconds is None or report['complete'] or time.time() >= config['deadline_epoch']: return
        time.sleep(min(a.watch_seconds, max(0, config['deadline_epoch']-time.time())))


if __name__ == '__main__': main()
