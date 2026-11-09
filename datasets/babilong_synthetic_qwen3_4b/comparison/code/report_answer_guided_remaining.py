"""Optional CPU-only posthoc quality snapshot; never called by generation.

This reads saved questions and evaluation-only supporting-fact positions. It
does not load models, inspect hidden answers, make digest checks, or write any
generation input. Lexical focus and source exposure are not semantic validity.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
import time

from contextualized.data import read_indexed_case
from experiments.analyze_synthetic_comparison import (
    TOKENS, intervals_cover, original_generation_geometry, query_terms, normalized)


VERSION = 'answer-guided-remaining-posthoc-v1'
DEFAULT_ROOT = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/answer_guided_api_remaining_20261108'
DEFAULT_ORIGINAL = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/contextual_complete_20261108'
METRICS = ('query_term_hit_proxy', 'context_union_support_recall_min',
           'context_union_support_recall_max', 'complete_context_fraction_min',
           'complete_context_fraction_max', 'has_complete_context_min',
           'has_complete_context_max', 'multisource_context_fraction',
           'all_support_quote_fraction_min', 'quoted_support_spans_multiple_original_chunks_fraction',
           'support_touching_context_fraction_min', 'support_touching_context_fraction_max',
           'no_support_context_fraction', 'exact_query_copy_fraction', 'exact_duplicate_fraction',
           'mean_question_words', 'mean_evidence_characters')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def average(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def original_index(roots, config, planned):
    """Fixed root order, plain original-recipe matching; never choose by quality."""
    expected = {'model': config['model'], 'train_samples': 64, 'validation_samples': 16,
        'chunk_tokens': 2048, 'question_tokens': 512, 'answer_tokens': 1024,
        'synthesis_batch': 4, 'seed': config.get('seed', 42)}
    result, issues = {}, []
    for value in roots:
        root = Path(value)
        options = [root, *sorted(root.glob('full/qwen/shard_*')), root/'integration/qwen']
        for directory in dict.fromkeys(options):
            run = read(directory/'run_config.json') if (directory/'run_config.json').is_file() else {}
            settings = run.get('settings', {})
            legacy_compatible = all(settings.get(k) == v for k, v in expected.items())
            for base in sorted((directory/'datasets').glob('*')):
                key = base.name
                if key not in planned or key in result:
                    continue
                plain = [base/'original_train.json', base/'original_validation.json']
                legacy = [base/'original'/split/'conversations.json' for split in ('train', 'validation')]
                try:
                    if all(p.is_file() for p in plain):
                        bodies = [read(p) for p in plain]
                        if any(body.get('settings') != expected for body in bodies):
                            continue
                        files, catalog = plain, base/'sources.json'
                    elif legacy_compatible and all(p.is_file() for p in legacy):
                        files, bodies = legacy, [read(p) for p in legacy]
                        catalog = Path(bodies[0]['context_catalog'])
                    else:
                        continue
                    if not catalog.is_file():
                        continue
                    if any(body.get('case') != key or body.get('actual_count') != count
                           or len(body.get('conversations', [])) != count
                           for body, count in zip(bodies, (64, 16))):
                        raise ValueError('Original split identity/count is incomplete')
                    result[key] = {'files': [str(p) for p in files], 'source': str(catalog),
                        'root': str(directory), 'recipe': expected,
                        'generation_provenance': [body.get('provenance') for body in bodies]}
                except (OSError, ValueError, KeyError, TypeError) as error:
                    issues.append({'case': key, 'path': str(base), 'error': str(error)})
    return result, issues


def contexts_from_original(source, document):
    contexts = {}
    for row in source.get('original_contexts', source.get('contexts', [])):
        if row.get('kind') != 'decoded_token_chunk':
            continue
        span = row.get('document_char_span')
        known = (isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span)
                 and 0 <= span[0] < span[1] <= len(document)
                 and document[span[0]:span[1]] == row.get('text'))
        contexts[row['context_id']] = {**row, 'source_spans': [span] if known else None}
    return contexts


def checked_facts(annotation, case):
    if annotation is None:
        return None
    if (annotation.get('case') != case['key'] or annotation.get('task') != case['task']
            or annotation.get('length') != case['length']
            or annotation.get('document_characters') != len(case['document'])):
        raise ValueError('Support annotation identity/length differs from the source case')
    facts = []
    for fact in annotation['facts']:
        if not fact['is_support']:
            continue
        places = fact.get('possible_document_spans', [])
        if not places or any(not isinstance(span, list) or len(span) != 2
                or any(type(x) is not int for x in span)
                or not 0 <= span[0] < span[1] <= len(case['document']) for span in places):
            raise ValueError('Support positions fall outside the source document')
        texts = [case['document'][a:b] for a, b in places]
        if len(set(texts)) != 1:
            raise ValueError('Ambiguous support positions refer to different source strings')
        facts.append((fact, places, texts[0]))
    if not facts:
        raise ValueError('Support annotation contains no supporting fact')
    return facts


def source_spans(example, document, contexts):
    evidence = example.get('evidence')
    if evidence is None:
        context = contexts.get(example.get('source_context_id'))
        return (context.get('source_spans'), None) if context else (None, None)
    chunks = evidence.get('source_chunks', [])
    spans = []
    for chunk in chunks:
        a, b = chunk['span']
        if (not 0 <= a < b <= len(document) or chunk.get('text') != document[a:b]
                or chunk['text'] not in evidence.get('text', '')):
            return None, None
        spans.append([a, b])
    if not spans:
        return None, None
    raw_quotes = example.get('answer_source_quotes')
    if not isinstance(raw_quotes, list) or not raw_quotes:
        return spans, None
    quoted = []
    for quote in raw_quotes:
        text = quote.get('quote') if isinstance(quote, dict) else quote
        if not isinstance(text, str) or not text:
            return spans, None
        found = []
        for chunk in chunks:
            offset = 0
            while (at := chunk['text'].find(text, offset)) >= 0:
                found.append([chunk['span'][0]+at, chunk['span'][0]+at+len(text)])
                offset = at+len(text)
        if not found:
            return spans, None
        quoted.extend(found)
    return spans, quoted


def coverage(spans, facts):
    if spans is None or facts is None:
        return None, None
    return (sum(all(intervals_cover(spans, p) for p in places) for _, places, _ in facts)/len(facts),
            sum(any(intervals_cover(spans, p) for p in places) for _, places, _ in facts)/len(facts))


def question_text(example):
    if isinstance(example.get('question'), str):
        return example['question']
    return next(message['content'] for message in example['messages'] if message['role'] == 'user')


def arm_metrics(examples, case, facts, contexts):
    terms = set(query_terms(case['query']))
    questions = [question_text(row) for row in examples]
    known, union, rows = True, [], []
    multisource, quote_complete, joint_quote, evidence_characters = [], [], [], []
    for example in examples:
        spans, quotes = source_spans(example, case['document'], contexts)
        known = known and spans is not None
        if spans is not None:
            union.extend(spans)
            merged = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]: merged[-1][1] = max(end, merged[-1][1])
                else: merged.append([start,end])
            evidence_characters.append(sum(end-start for start,end in merged))
        lo, hi = coverage(spans, facts)
        rows.append((lo, hi))
        if spans is not None:
            multisource.append(float(len(set(map(tuple, spans))) > 1))
        if quotes is not None and facts is not None:
            quoted_facts = [item for item in facts if all(intervals_cover(quotes, p) for p in item[1])]
            quote_complete.append(float(len(quoted_facts) == len(facts)))
            geometry = original_generation_geometry(quoted_facts, contexts) if quoted_facts else None
            joint_quote.append(geometry['multi_fact_requires_multiple_original_generation_contexts']
                               if geometry else False)
    union_min, union_max = coverage(union if known else None, facts)
    lows, highs = [r[0] for r in rows if r[0] is not None], [r[1] for r in rows if r[1] is not None]
    # Empty finished datasets have zero union coverage; per-question rates are
    # unavailable, since no question was produced. Both facts are retained.
    complete_min = bool(lows and max(lows) == 1) if known and facts is not None else None
    complete_max = bool(highs and max(highs) == 1) if known and facts is not None else None
    return {'questions': len(examples), 'empty_dataset': not examples,
        'query_term_hit_proxy': (sum(bool(terms & set(TOKENS.findall(q.casefold()))) for q in questions)/len(questions)
                                 if terms and questions else None),
        'exact_query_copy_count': sum(normalized(q) == normalized(case['query']) for q in questions),
        'exact_duplicate_count': len(questions)-len({normalized(q) for q in questions}),
        'exact_query_copy_fraction': (sum(normalized(q)==normalized(case['query']) for q in questions)/len(questions)
                                      if questions else None),
        'exact_duplicate_fraction': ((len(questions)-len({normalized(q) for q in questions}))/len(questions)
                                    if questions else None),
        'mean_question_words': average([len(TOKENS.findall(q)) for q in questions]),
        'mean_evidence_characters': average(evidence_characters),
        'context_union_support_recall_min': union_min, 'context_union_support_recall_max': union_max,
        'complete_context_fraction_min': average([float(value == 1) for value in lows]),
        'complete_context_fraction_max': average([float(value == 1) for value in highs]),
        'support_touching_context_fraction_min': average([float(value>0) for value in lows]),
        'support_touching_context_fraction_max': average([float(value>0) for value in highs]),
        'no_support_context_fraction': average([float(value==0) for value in highs]),
        'has_complete_context_min': complete_min, 'has_complete_context_max': complete_max,
        'context_position_known_questions': len(lows) if facts is not None else sum(source_spans(e, case['document'], contexts)[0] is not None for e in examples),
        'multisource_context_fraction': average(multisource),
        'all_support_quote_fraction_min': average(quote_complete),
        'quote_eligible_questions': len(quote_complete),
        'quoted_support_spans_multiple_original_chunks_fraction': average(joint_quote),
        'joint_quote_geometry_eligible_questions': sum(x is not None for x in joint_quote),
        'semantic_relevance_or_answer_correctness': 'not_assessed'}


def aggregate(rows):
    result = {'cases': len(rows), 'unique_source_questions': len({row['sampling_unit'] for row in rows
                if row.get('sampling_unit') is not None}), 'arms': {}, 'paired_differences': {}}
    for arm in ('original', 'answer_guided'):
        available = [row['arms'][arm] for row in rows if arm in row['arms']]
        result['arms'][arm] = {'available_cases': len(available),
            'questions': sum(item['questions'] for item in available),
            'zero_question_cases': sum(item['empty_dataset'] for item in available),
            'questions_per_case': average([item['questions'] for item in available]),
            'metrics': {name: {'macro_mean': average([item[name] for item in available]),
                'eligible_cases': sum(item[name] is not None for item in available)} for name in METRICS}}
    paired = [row for row in rows if set(row['arms']) == {'original', 'answer_guided'}]
    result['paired_cases'] = len(paired)
    for name in METRICS:
        pairs = [(row['arms']['original'][name], row['arms']['answer_guided'][name]) for row in paired
                 if row['arms']['original'][name] is not None and row['arms']['answer_guided'][name] is not None]
        result['paired_differences'][name] = {'eligible_paired_cases': len(pairs),
            'original_macro_mean': average([a for a, _ in pairs]),
            'guided_macro_mean': average([b for _, b in pairs]),
            'guided_minus_original': average([b-a for a, b in pairs])}
    return result


def timing_summary(records, now, config, issues):
    inherited = [r for r in records if r['inherited']]
    fresh = [r for r in records if not r['inherited']]
    totals = {kind: {'cases': 0, 'stored_calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
                     'historical_or_request_wall_seconds_sum': 0., 'unknown_token_usage_calls': 0}
              for kind in ('fresh', 'inherited')}
    cached_api = {'reused_case_records': 0, 'historical_scoring_seconds_known_sum': 0.,
                  'historical_estimated_cost_usd_known_sum': 0., 'new_api_requests': 0}
    for record in records:
        kind = 'inherited' if record['inherited'] else 'fresh'
        total = totals[kind]; total['cases'] += 1
        try:
            checkpoint = Path(record['paths'].get('calls', '/missing'))
            if kind == 'fresh' and checkpoint.is_file():
                calls = read(checkpoint)['calls']
                values = [dict(row['response'], wall_seconds=row.get('wall_seconds')) for row in calls]
            else:
                values = read(record['paths']['audit']).get('calls', [])
            total['stored_calls'] += len(values)
            for value in values:
                for name in ('prompt_tokens', 'completion_tokens'):
                    if isinstance(value.get(name), (int, float)):
                        total[name] += value[name]
                total['unknown_token_usage_calls'] += any(not isinstance(value.get(name), (int, float))
                    for name in ('prompt_tokens', 'completion_tokens'))
                total['historical_or_request_wall_seconds_sum'] += value.get('wall_seconds') or 0
            if kind == 'fresh':
                source = read(record['paths']['source'])
                path = source.get('decision_score_file')
                if path and Path(path).is_file():
                    cache = read(path); cached_api['reused_case_records'] += 1
                    cached_api['historical_scoring_seconds_known_sum'] += cache.get('scoring_seconds') or 0
                    cached_api['historical_estimated_cost_usd_known_sum'] += cache.get('estimated_cost_usd') or 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append({'case': record['case'], 'stage': 'timing', 'error': str(error)})
    began = datetime.fromisoformat(config['started_utc']).timestamp() if config.get('started_utc') else None
    elapsed = max(0, now-began) if began is not None else None
    return {'cost_by_provenance': totals, 'decision_api_cache': cached_api,
        'completed_fresh_cases': len(fresh), 'inherited_cases': len(inherited),
        'fresh_case_elapsed_seconds_mean': average([r.get('timings', {}).get('case_seconds') for r in fresh]),
        'campaign_elapsed_seconds_including_queue': elapsed,
        'completed_fresh_cases_per_campaign_hour': len(fresh)*3600/elapsed if elapsed else None,
        'completed_fresh_questions_per_campaign_hour': sum(r['counts']['answer_guided'] for r in fresh)*3600/elapsed if elapsed else None,
        'completed_fresh_output_tokens_per_campaign_second': totals['fresh']['completion_tokens']/elapsed if elapsed else None,
        'original_generation_in_this_campaign': False,
        'caveats': ['Fresh totals cover completed cases; in-flight work is not included.',
            'Per-call and per-case times overlap across asynchronous cases and GPUs; their sum is not wall time or GPU compute.',
            'Inherited data and cached Decision API costs are historical, not new generation cost.',
            'Campaign throughput includes queueing and startup; it is not isolated decoder throughput.',
            'Different backends/concurrency and unknown legacy original costs prevent a fair original-versus-guided speed claim.']}


def build_report(root, *, original_roots=None, annotations_path=None, vanilla_root=None):
    root = Path(root); config = read(root/'config.json'); plan = read(config['plan'])
    planned = {row['key']: row for row in plan['cases']}
    index = {row['key']: row for row in read(plan['source_index'])}
    roots = original_roots or config.get('report_original_roots') or [DEFAULT_ORIGINAL, *config.get('reuse_roots', [])]
    if vanilla_root is None:
        originals, issues = original_index(roots, config, planned)
    else:
        originals, issues = {}, []
        for key in planned:
            marker = read(Path(vanilla_root)/'dataset_cases'/(key+'.json'))
            specific = read(root/'dataset_cases'/(key+'.json'))
            expected={split:specific['counts'][split] for split in ('train','validation')}
            if marker.get('complete') is not True or marker['counts'] != expected:
                raise ValueError('Matched vanilla case missing or counts differ: '+key)
            originals[key]={'source':marker['paths']['source'],
                'files':[marker['paths']['original_train'],marker['paths']['original_validation']],
                'root':str(vanilla_root),'recipe':'official_matched_count_fresh_generation'}
    annotations_path = Path(annotations_path or config.get('report_support_annotations') or
                            Path(config['plan']).parent/'evaluation/source_annotations.jsonl')
    annotations = {}
    if annotations_path.is_file():
        for line in annotations_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record['case'] in annotations:
                    raise ValueError('Duplicate supporting-fact annotation')
                annotations[record['case']] = record
    rows, markers, counts = [], [], Counter()
    for path in sorted((root/'dataset_cases').glob('*.json')):
        try:
            marker = read(path); key = marker['case']
            if key not in planned or marker.get('complete') is not True:
                continue
            counts['completed_markers'] += 1
            case = {**read_indexed_case(plan, index, key),
                    'task': planned[key]['task'], 'length': planned[key]['length']}
            if marker.get('query') != case['query']:
                raise ValueError('Completed dataset query differs from its source case')
            facts = checked_facts(annotations.get(key), case)
            original, contexts = None, {}
            if key in originals:
                baseline = originals[key]
                contexts = contexts_from_original(read(baseline['source']), case['document'])
                original = [row for file in baseline['files'] for row in read(file)['conversations']]
            guided, split_counts = [], {}
            for split in ('train', 'validation'):
                body = read(marker['paths'][split]); examples = body['examples']
                if (body.get('case') != key or body.get('query') != case['query']
                        or body.get('actual_count') != len(examples)
                        or marker['counts'][split] != len(examples)):
                    raise ValueError('Guided split identity/count is inconsistent')
                guided.extend(examples); split_counts[split] = len(examples)
            if len(guided) != marker['counts']['answer_guided']:
                raise ValueError('Guided total differs from its saved splits')
            geometry = original_generation_geometry(facts, contexts) if facts and contexts else None
            row = {'case': key, 'task': case['task'], 'length': case['length'],
                'sampling_unit': annotations.get(key, {}).get('sampling_unit'),
                'episode_cluster': annotations.get(key, {}).get('original_episode_component'),
                'support_count': len(facts) if facts else None,
                'ambiguous_support_positions': any(len(item[1]) > 1 for item in facts) if facts else None,
                'provenance_group': 'inherited' if marker['inherited'] else 'fresh',
                'guided_counts': split_counts, 'guided_target': 32,
                'guided_shortfall': max(0, 32-len(guided)),
                'original_source': originals.get(key), 'guided_marker': str(path),
                'support_requires_multiple_original_chunks': geometry['multi_fact_requires_multiple_original_generation_contexts'] if geometry else None,
                'arms': {'answer_guided': arm_metrics(guided, case, facts, contexts)}}
            if original is not None:
                row['arms']['original'] = arm_metrics(original, case, facts, contexts)
            rows.append(row); markers.append(marker)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            issues.append({'path': str(path), 'stage': 'case_analysis', 'error': str(error)})
    now = time.time(); grouped = {}
    for provenance in ('inherited', 'fresh'):
        part = [row for row in rows if row['provenance_group'] == provenance]
        grouped[provenance] = aggregate(part)
    failures = [p.stem for p in (root/'dataset_failures').glob('*.json')]
    report = {'schema': VERSION, 'created_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
        'root': str(root), 'planned_cases': len(planned), 'compatible_original_cases': len(originals),
        'compatible_original_case_ids': sorted(originals), 'completed_guided_markers': counts['completed_markers'],
        'analyzed_guided_cases': len(rows), 'unfinished_cases': len(planned)-counts['completed_markers'],
        'reported_failure_case_ids': sorted(set(failures)&set(planned)),
        'guided_questions': sum(row['arms']['answer_guided']['questions'] for row in rows),
        'guided_total_shortfall_over_finished_cases': sum(row['guided_shortfall'] for row in rows),
        'guided_finished_cases_with_shortfall': sum(row['guided_shortfall'] > 0 for row in rows),
        'source_annotation_path': str(annotations_path) if annotations_path.is_file() else None,
        'analysis_only': True, 'model_calls': 0, 'training_performed': False,
        'annotations_passed_to_generation': False, 'benchmark_answer_values_read': False,
        'overall': aggregate(rows), 'by_provenance': grouped,
        'multi_chunk_support_subset': aggregate([row for row in rows if row['support_requires_multiple_original_chunks'] is True]),
        'by_length': {str(length): aggregate([row for row in rows if row['length'] == length]) for length in (8192,16384,32768,65536)},
        'timing': timing_summary(markers, now, config, issues), 'issues': issues, 'cases': rows,
        'limitations': ['Query-term hits are a lexical focus proxy, not semantic relevance or correctness.',
            'Coverage concerns official support for the real query, not whether each generated question has a correct answer.',
            'Context/quote co-exposure across chunks does not prove that a question requires or correctly combines those facts.',
            'Original conversations do not provide source quotes; missing quote metrics are unavailable, not zero quality.',
            'Paired comparisons use only the same completed cases with compatible original data; completion order is not random.',
            'Inherited legacy lexical/model-cluster data and fresh cached-Decision-API/deterministic-cluster data are separate protocols.',
            ('Per-case train/validation question counts are matched between the two datasets.' if vanilla_root else
             'Original has 80 questions per case; Guided has at most 32. Union coverage is not a matched-question-budget experiment.'),
            'The 1000 length/task cases derive from 250 source questions; cases are not independent.',
            'Ambiguous support locations retain lower/upper bounds; unavailable source geometry is not called fragmented.',
            'This optional offline report is not a generation check, admission gate, retry loop, or proof of downstream benefit.']}
    report['matched_vanilla_root']=str(vanilla_root) if vanilla_root else None
    report['question_budgets_matched_per_case']=vanilla_root is not None
    return report


def markdown(report):
    lines = [f"Saved dataset snapshot: {report['created_utc']}", '',
        f"Guided: {report['analyzed_guided_cases']}/{report['planned_cases']} cases, {report['guided_questions']} questions; "
        f"compatible original: {report['compatible_original_cases']} cases; paired: {report['overall']['paired_cases']}.", '',
        '| Provenance | Paired cases | Metric | Original | Guided |',
        '|---|---:|---|---:|---:|']
    labels = {'query_term_hit_proxy': 'Query-term hit proxy',
        'context_union_support_recall_min': 'Support recall, lower bound',
        'complete_context_fraction_min': 'All-support context fraction, lower bound'}
    def percent(value):
        return '—' if value is None else f'{100*value:.2f}%'
    for group, summary in report['by_provenance'].items():
        for metric, label in labels.items():
            data = summary['paired_differences'][metric]
            lines.append(f"| {group} | {data['eligible_paired_cases']} | {label} | "
                f"{percent(data['original_macro_mean'])} | {percent(data['guided_macro_mean'])} |")
    lines += ['', f"Finished-case question shortfall: {report['guided_total_shortfall_over_finished_cases']} "
              f"across {report['guided_finished_cases_with_shortfall']} cases; analysis issues: {len(report['issues'])}.", '',
              'These are lexical-focus and source-exposure diagnostics. They do not establish semantic correctness, '
              'necessary cross-chunk reasoning, or an improvement in Cartridge accuracy.', '',
              'Request timings overlap under concurrency. Historical data/cache costs are separate from fresh generation.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--original-root', action='append', default=None)
    parser.add_argument('--vanilla-root', help='Fresh vanilla dataset with the same per-case split counts')
    parser.add_argument('--support-annotations')
    parser.add_argument('--output', help='Defaults to <root>/posthoc_quality; never modifies generation inputs')
    args = parser.parse_args()
    report = build_report(args.root, original_roots=args.original_root,
                          annotations_path=args.support_annotations, vanilla_root=args.vanilla_root)
    output = Path(args.output) if args.output else Path(args.root)/'posthoc_quality'
    write(output/'report.json', report)
    temporary = output/f'.summary.{os.getpid()}.tmp'
    temporary.write_text(markdown(report)); temporary.replace(output/'summary.md')
    print(json.dumps({'paired_cases': report['overall']['paired_cases'],
        'guided_cases': report['analyzed_guided_cases'], 'guided_questions': report['guided_questions'],
        'analysis_issues': len(report['issues']), 'report': str(output/'report.json')}))


if __name__ == '__main__':
    main()
