"""Finish the fixed BABILong Answer-Guided dataset; no baseline or KV training.

All workers share a nonblocking, per-case flock queue. Completed old triples
remain inherited data with their original protocol, counts and source records.
"""
import argparse
import asyncio
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback
from types import SimpleNamespace
import zlib

from contextualized import answer_guided_questions as A
from contextualized.data import read_indexed_case
from experiments import build_synthetic_comparison as B
from experiments import build_synthetic_pair_fast as F


DEFAULT_VERSION = 'answer-guided-remaining-v3'


class DeadlineReached(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def recipe(config):
    return {'version': config.get('generation_worker_version', DEFAULT_VERSION),
            'model': config['model'], 'seed': config.get('seed', 42),
            'generator': 'contextualized.answer_guided_questions.build_answer_guided_dataset',
            'answer_tokens': config.get('answer_tokens', 1024),
            'question_tokens': config.get('question_tokens', 512),
            'questions_per_answer': config.get('questions_per_answer', 4),
            'max_examples': config.get('max_examples', 32),
            'train_samples': config.get('guided_train_samples', 24),
            'validation_samples': config.get('guided_validation_samples', 8),
            'candidate_pool': config.get('candidate_pool', 16),
            'max_clusters': config.get('max_clusters', 32),
            'max_chunks': config.get('max_cluster_chunks', 8),
            'support_tokens': config.get('support_tokens', 32768),
            'simple_two_stage': True,
            'evidence_policy': 'cached-decision-api-with-source-recall-safeguards',
            'sampling_protocol': 'case-call-local-v1/vllm-request-seed-v1',
            'original_generation': False, 'cartridge_training': False}


def paths(output, key):
    base = Path(output) / 'datasets' / key
    return {'source': str(base / 'sources.json'), 'clusters': str(base / 'clusters.json'),
            'bundle': str(base / 'answer_guided.json'), 'audit': str(base / 'audit.json'),
            'calls': str(base / 'calls_checkpoint.json'),
            'train': str(base / 'answer_guided/train/examples.json'),
            'validation': str(base / 'answer_guided/validation/examples.json')}


def case_marker(output, key):
    return Path(output) / 'dataset_cases' / (key + '.json')


def completed(output, key):
    marker = case_marker(output, key)
    if not marker.is_file():
        return False
    try:
        record = read(marker)
    except FileNotFoundError:
        # A different worker may just have archived an old empty attempt.
        return False
    if record.get('case') != key or record.get('complete') is not True:
        raise ValueError(f'Invalid completion record: {marker}')
    if (record.get('counts', {}).get('answer_guided') == 0
            and not record.get('provenance', {}).get('empty_repair_attempted')):
        return False
    for name in ('source', 'clusters', 'bundle', 'audit', 'train', 'validation'):
        if not Path(record['paths'][name]).is_file():
            raise ValueError(f'Completed case is missing {name}: {key}')
    return True


def claim(output, key):
    directory = Path(output) / 'claims'
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / (key + '.lock')).open('a+')
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def release(stream):
    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    stream.close()


def archive_incompatible_attempt(state, key):
    """Called only while holding the case flock; retain every old artifact."""
    output = Path(state.config['output'])
    marker = case_marker(output, key)
    saved = paths(output, key)
    reasons = []
    if marker.is_file():
        old = read(marker)
        if old.get('counts', {}).get('answer_guided', 0) > 0:
            return
        if not old.get('provenance', {}).get('empty_repair_attempted'):
            reasons.append('empty_before_decision_cluster_recovery')
            state.empty_repair_cases.add(key)
        if old.get('settings', {}).get('version') != state.settings['version']:
            reasons.append('empty_completion_from_previous_generation_policy')
    for name in ('source', 'clusters', 'calls'):
        path = Path(saved[name])
        if path.is_file() and read(path).get('settings') != state.settings:
            reasons.append(name+'_checkpoint_from_previous_generation_policy')
    if not reasons:
        return
    destination = output/'attempts'/key/f'legacy_generation_{time.time_ns()}_{os.getpid()}'
    destination.mkdir(parents=True, exist_ok=False)
    moved = {}
    if marker.is_file():
        target = destination/'legacy_empty_case.json'
        marker.replace(target)
        moved[str(marker)] = str(target)
    base = output/'datasets'/key
    if base.exists():
        target = destination/'datasets'
        base.replace(target)
        moved[str(base)] = str(target)
    failure = output/'dataset_failures'/(key+'.json')
    if failure.exists():
        target = destination/'legacy_failure.json'
        failure.replace(target)
        moved[str(failure)] = str(target)
    write(destination/'archive.json', {'case': key, 'reasons': reasons,
        'moved_paths': moved, 'new_settings': state.settings, 'epoch': time.time(),
        'old_records_preserved_without_relabeling': True})


def old_run_roots(roots):
    result = []
    for value in roots:
        root = Path(value)
        options = [root, *sorted(root.glob('full/qwen/shard_*')), root / 'integration/qwen']
        for option in options:
            if (option / 'run_config.json').is_file() and option not in result:
                result.append(option)
    return result


def inherited_index(config, planned):
    """Keep the first recorded legacy version of each case; never pick by score."""
    result = {}
    expected = recipe(config)
    for root in old_run_roots(config.get('reuse_roots', [])):
        settings = read(root / 'run_config.json').get('settings', {})
        # A v2 dataset with eight paraphrases per answer is not interchangeable
        # with the v3 evidence-anchored/four-variant protocol.  Reuse only an
        # artifact produced by the exact current recipe, never merely because
        # it has the same model and nominal arm name.
        if any(settings.get(name) != value for name, value in expected.items()):
            continue
        for marker in sorted((root / 'cases').glob('*.json')):
            record = read(marker)
            key = record.get('case')
            if key not in planned or key in result or record.get('data_available') is not True:
                continue
            arm = record.get('arms', {}).get('answer_guided', {})
            if not all(split in arm for split in ('train', 'validation')):
                continue
            if not all(Path(arm[split].get('json_path', '/missing')).is_file()
                       for split in ('train', 'validation')):
                continue
            result[key] = {'marker': str(marker), 'record': record, 'settings': settings}
    return result


def check_sources(source_chunks, case):
    for chunk in source_chunks:
        start, end = chunk['span']
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(case['document'])
                or chunk.get('text') != case['document'][start:end]):
            raise ValueError('Saved evidence does not match its original source text')


def check_examples(examples, case):
    for item in examples:
        if not all(isinstance(item.get(name), str) and item[name].strip()
                   for name in ('question', 'answer')):
            raise ValueError('Answer-Guided record lacks a question or answer')
        evidence = item['evidence']
        check_sources(evidence['source_chunks'], case)
        if any(chunk['text'] not in evidence['text'] for chunk in evidence['source_chunks']):
            raise ValueError('Evidence cluster omits its recorded source text')


def publish(state, case, grouped, bundle, provenance, timings, usage, *, inherited=False):
    output, key = state.config['output'], case['key']
    saved = paths(output, key)
    train_samples = state.settings['train_samples']
    validation_samples = state.settings['validation_samples']
    counts = {split: len(grouped[split]) for split in ('train', 'validation')}
    counts['answer_guided'] = counts['train'] + counts['validation']
    if counts['train'] > train_samples or counts['validation'] > validation_samples:
        raise ValueError('Saved Answer-Guided splits exceed the unchanged budgets')
    for split, requested in (('train', train_samples), ('validation', validation_samples)):
        check_examples(grouped[split], case)
        write(saved[split], {'schema': 'synthetic-comparison-triples-v2', 'case': key,
            'query': case['query'], 'arm': 'answer_guided', 'split': split,
            'requested_count': requested, 'examples': grouped[split],
            'actual_count': counts[split], 'shortfall': max(0, requested-counts[split]),
            'target_count_met': counts[split] == requested,
            'document_source': {'path': state.plan['source_cases'],
                                **{name: state.index[key][name] for name in ('offset', 'bytes')}},
            'context_catalog': saved['source'], 'provenance': provenance,
            'teacher_forced_targets_created': False, 'training_performed': False,
            'actual_query_anchor_added': False, 'benchmark_answers_read': False,
            'source_facts_verified': False})
    write(saved['bundle'], bundle)
    write(saved['audit'], bundle['audit'])
    write(case_marker(output, key), {'schema': state.settings['version'], 'case': key, 'query': case['query'],
        'task': case['task'], 'length': case['length'], 'complete': True,
        'complete_means': 'generation finished and all artifacts saved; not sample budget or quality guaranteed',
        'data_available': counts['answer_guided'] > 0, 'counts': counts,
        'requested_counts': {'train': train_samples, 'validation': validation_samples,
                             'answer_guided': train_samples + validation_samples},
        'shortfalls': {'train': train_samples-counts['train'],
                       'validation': validation_samples-counts['validation']},
        'all_target_counts_met': (counts['train'] == train_samples
                                  and counts['validation'] == validation_samples),
        'settings': state.settings, 'inherited': inherited, 'provenance': provenance,
        'paths': saved, 'timings': timings, 'tokens_generated_this_attempt': usage,
        'benchmark_answers_used': False, 'semantic_quality_verified': False,
        'cartridge_training': False, 'original_generation': False,
        'worker_rank': state.rank, 'completed_epoch': time.time()})


def import_case(state, case, previous):
    record = previous['record']
    if record.get('query') != case['query']:
        raise ValueError('Inherited case has a different query')
    grouped = {}
    for split in ('train', 'validation'):
        source = read(record['arms']['answer_guided'][split]['json_path'])
        if source.get('case') != case['key'] or source.get('query') != case['query']:
            raise ValueError('Inherited split belongs to another case')
        grouped[split] = source['examples']
        if source.get('actual_count') != len(grouped[split]):
            raise ValueError('Inherited split count differs from its saved records')
    raw = read(record['answer_guided_bundle_path'])
    bundle = raw.get('content', raw)
    source = read(record['context_catalog'])
    check_sources(source['source_chunks'], case)
    saved = paths(state.config['output'], case['key'])
    write(saved['source'], source)
    write(saved['clusters'], read(record['clusters_path']))
    provenance = {'kind': 'inherited_answer_guided', 'source_case_manifest': previous['marker'],
        'source_split_paths': {split: record['arms']['answer_guided'][split]['json_path']
                               for split in ('train', 'validation')},
        'source_settings': previous['settings'], 'source_retrieval': source.get('retrieval'),
        'generation_protocol_changed': False,
        'note': 'Legacy two-stage Qwen4B/lexical-evidence data retained as originally generated; not Decision API generation.'}
    publish(state, case, grouped, bundle, provenance, {'inherited_generation_seconds': None},
            {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0}, inherited=True)


def checkpoint_identity(record, case, settings):
    if record.get('case') != case['key'] or record.get('query') != case['query'] or record.get('settings') != settings:
        raise ValueError('Checkpoint case/query/settings mismatch')


async def generate_case(state, case):
    from experiments.decision_answer_evidence import prepare_evidence
    from experiments.vllm_study_client import sampling_seed
    started = time.monotonic()
    saved = paths(state.config['output'], case['key'])
    if Path(saved['source']).exists() and Path(saved['clusters']).exists():
        source, groups = read(saved['source']), read(saved['clusters'])
        checkpoint_identity(source, case, state.settings)
        checkpoint_identity(groups, case, state.settings)
        check_sources(source['source_catalog']['source_chunks'], case)
        clusters = groups['clusters']
    else:
        score_path = Path(state.plan['shared_scores']) / 'scores' / state.mapping[case['key']]
        prepared = await prepare_evidence(case, read(score_path), state.engine.tokenizer,
            candidate_pool=state.settings['candidate_pool'],
            max_clusters=state.settings['max_clusters'],
            max_chunks=state.settings['max_chunks'],
            support_tokens=state.settings['support_tokens'])
        source = {'case': case['key'], 'query': case['query'], 'settings': state.settings,
                  'source_catalog': prepared['source_catalog'], 'decision_score_file': str(score_path)}
        clusters = prepared['clusters']
        groups = {'case': case['key'], 'query': case['query'], 'settings': state.settings,
                  'clusters': clusters, 'audit': prepared['audit']}
        write(saved['source'], source)
        write(saved['clusters'], groups)
    calls = []
    if Path(saved['calls']).is_file():
        checkpoint = read(saved['calls'])
        checkpoint_identity(checkpoint, case, state.settings)
        calls = checkpoint['calls']
    cursor = 0
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0}
    replayed = 0
    seed = (state.config.get('seed', 42) + zlib.crc32(case['key'].encode())) % (2**31)

    async def call(engine, instruction, payload, *, max_tokens):
        nonlocal cursor, replayed
        if state.should_stop():
            raise DeadlineReached()
        signature = {'instruction': instruction, 'payload': payload, 'max_tokens': max_tokens}
        position = cursor
        cursor += 1
        if position < len(calls):
            if calls[position]['request'] != signature:
                raise ValueError('Saved call differs from deterministic replay request')
            replayed += 1
            return calls[position]['response']
        call_started = time.monotonic()
        with sampling_seed((seed+position) % (2**31)):
            response = await state.call(engine, instruction, payload, max_tokens=max_tokens)
        calls.append({'request': signature, 'response': response,
                      'wall_seconds': time.monotonic()-call_started})
        for name in ('prompt_tokens', 'completion_tokens'):
            usage[name] += response.get(name, 0) or 0
        usage['calls'] += 1
        write(saved['calls'], {'case': case['key'], 'query': case['query'],
            'settings': state.settings, 'calls': calls, 'generation_finished': False,
            'checkpoint_epoch': time.time()})
        return response

    bundle = await A.build_answer_guided_dataset(state.engine, case['query'], clusters,
        max_examples=state.settings['max_examples'],
        answer_tokens=state.settings['answer_tokens'],
        question_tokens=state.settings['question_tokens'],
        questions_per_answer=state.settings['questions_per_answer'],
        call_model=call, simple_two_stage=True)
    bundle['audit']['replayed_checkpoint_calls'] = replayed
    bundle['audit']['call_timing_scope'] = (
        'This attempt, including batch queue waits. Original durations of replayed calls remain in calls_checkpoint.json.')
    grouped = B.split_examples(bundle['examples'], state.settings['train_samples'],
                               state.settings['validation_samples'])
    provenance = {'kind': 'generated_answer_first_frozen_model',
        'empty_repair_attempted': case['key'] in state.empty_repair_cases,
        'backend': state.engine.backend_metadata, 'source_evidence': groups['audit'],
        'replayed_checkpoint_calls': replayed, 'call_checkpoint': saved['calls'],
        'core_answer_first_question_second_preserved': True,
        'exact_generation_policy_unchanged': False, 'simple_two_stage': True,
        'new_backend_bitwise_equivalence_claimed': False}
    publish(state, case, grouped, bundle, provenance,
        {'case_seconds': time.monotonic()-started,
         'wall_time_includes_waiting_for_shared_concurrent_batches': True}, usage)
    return usage


def record_failure(state, key, error):
    write(Path(state.config['output'])/'dataset_failures'/(key+'.json'),
        {'case': key, 'rank': state.rank, 'error_type': type(error).__name__, 'error': str(error),
         'traceback': traceback.format_exc(), 'epoch': time.time(), 'automatic_retry': False})
    print(json.dumps({'event': 'answer_guided_failed', 'case': key, 'error': str(error)}), flush=True)


async def process_queue(state, items):
    queue = asyncio.Queue()
    for row in items:
        queue.put_nowait(row)
    totals = {'completed': 0, 'inherited': 0, 'failed': 0, 'claimed_elsewhere': 0,
              'already_complete': 0, 'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0}

    async def work():
        while not state.should_stop():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            key = item['key']
            lock = claim(state.config['output'], key)
            if lock is None:
                totals['claimed_elsewhere'] += 1
                continue
            try:
                if completed(state.config['output'], key):
                    totals['already_complete'] += 1
                    continue
                archive_incompatible_attempt(state, key)
                if (Path(state.config['output'])/'dataset_failures'/(key+'.json')).exists():
                    continue
                if state.should_stop():
                    return
                case = {**read_indexed_case(state.plan, state.index, key),
                        'task': item['task'], 'length': item['length']}
                if key in state.inherited:
                    import_case(state, case, state.inherited[key])
                    totals['inherited'] += 1
                else:
                    usage = await generate_case(state, case)
                    for name in ('calls', 'prompt_tokens', 'completion_tokens'):
                        totals[name] += usage[name]
                totals['completed'] += 1
                print(json.dumps({'event': 'answer_guided_complete', 'case': key, 'rank': state.rank}), flush=True)
            except DeadlineReached:
                return
            except Exception as error:
                totals['failed'] += 1
                record_failure(state, key, error)
                if state.engine is not None and state.engine.batcher.failure is not None:
                    state.stop['value'] = True
                    return
            finally:
                release(lock)
    await asyncio.gather(*(work() for _ in range(state.config.get('concurrent_cases', 16))))
    return totals


def run(config, rank):
    started = time.monotonic()
    plan = read(config['plan'])
    counts = Counter((row['length'], row['task']) for row in plan['cases'])
    expected = {(length, 'qa'+str(task)): 25 for length in (8192, 16384, 32768, 65536)
                for task in range(1, 11)}
    if len({row['key'] for row in plan['cases']}) != 1000 or counts != expected:
        raise ValueError('Require the unchanged 1000-case BABILong plan')
    if not 1 <= config.get('concurrent_cases', 16) <= 32:
        raise ValueError('Concurrent cases must be between 1 and 32')
    deadline = min(float(config['deadline_epoch']),
                   float(os.environ.get('PAIR_DEADLINE_EPOCH', config['deadline_epoch'])))
    stop = {'value': False}
    def on_signal(*_):
        stop['value'] = True
    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    def should_stop():
        return stop['value'] or time.time() >= deadline
    worker_path = Path(config['output'])/'dataset_workers'/f'rank_{rank}.json'
    def hard_stop():
        try:
            write(worker_path, {'rank': rank, 'status': 'deadline_checkpoint',
                  'wall_seconds': time.monotonic()-started, 'epoch': time.time(),
                  'note': 'Atomic per-call checkpoints remain; in-flight work may be incomplete.'})
        finally:
            os._exit(75)
    if should_stop():
        return 75
    timer = threading.Timer(max(0, deadline-time.time()), hard_stop)
    timer.daemon = True
    timer.start()
    state = SimpleNamespace(config=config, rank=rank, plan=plan, settings=recipe(config),
        index={row['key']: row for row in read(plan['source_index'])},
        mapping=read(Path(plan['shared_scores'])/'case_to_score_file.json'),
        inherited=inherited_index(config, {row['key'] for row in plan['cases']}),
        engine=None, should_stop=should_stop, stop=stop, empty_repair_cases=set())
    write(worker_path, {'rank': rank, 'status': 'starting', 'epoch': time.time(),
        'settings': state.settings, 'deadline_epoch': deadline, 'pid': os.getpid(),
        'job_id': os.environ.get('SLURM_JOB_ID'), 'static_sharding': False})
    components = None
    try:
        # Copy reusable data before loading any model, under the same case locks.
        reused = asyncio.run(process_queue(state, [row for row in plan['cases'] if row['key'] in state.inherited]))
        remaining = [row for row in plan['cases'] if not completed(config['output'], row['key'])]
        if remaining and not should_stop():
            from experiments.vllm_study_client import load_vllm_runtime
            state.engine = load_vllm_runtime(config['model'], seed=config.get('seed', 42),
                max_batch_sequences=config.get('max_batch_sequences', 128), batch_wait_seconds=.01,
                max_model_len=config.get('max_model_len', 32768),
                gpu_memory_utilization=config.get('gpu_memory_utilization', .9),
                max_num_batched_tokens=config.get('max_num_batched_tokens', 8192))
            components = F.make_runtime_components(state.engine)
            state.call = F.frozen_call(state.engine, components)
            generated = asyncio.run(process_queue(state, remaining))
        else:
            generated = {}
        completed_count = sum(completed(config['output'], row['key']) for row in plan['cases'])
        failed_count = sum((Path(config['output'])/'dataset_failures'/(row['key']+'.json')).exists()
                           for row in plan['cases'])
        write(worker_path, {'rank': rank, 'status': 'deadline_checkpoint' if should_stop() else 'queue_scan_finished',
            'epoch': time.time(), 'wall_seconds': time.monotonic()-started,
            'inherited_pass': reused, 'generated_pass': generated,
            'global_completed_cases': completed_count, 'global_failed_cases': failed_count,
            'global_remaining_cases': 1000-completed_count, 'full_dataset_complete': completed_count == 1000,
            'cartridge_training': False, 'original_generation': False})
        return 0 if completed_count == 1000 else (75 if should_stop() else (1 if failed_count else 0))
    finally:
        timer.cancel()
        if components is not None:
            from contextualized.upstream_study import _RUNTIMES
            _RUNTIMES.pop(components.runtime_key, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--rank', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    return run(read(args.config), args.rank)


if __name__ == '__main__':
    raise SystemExit(main())
