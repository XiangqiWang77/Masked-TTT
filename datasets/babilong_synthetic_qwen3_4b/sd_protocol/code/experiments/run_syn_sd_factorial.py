"""Paired ordinary/selective SD on the explicitly configured synthetic dataset."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback
import zlib


class TeacherPreparationError(RuntimeError):
    """A failed inference process must stop its worker instead of excluding more cases."""


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def input_rows(item, config, seed):
    """Use saved questions and source bindings; never read synthetic answer labels."""
    frozen = Path(config.get('paired_input_dir', config['output']+'/paired_inputs'))/(item['key']+'.json')
    if frozen.is_file():
        value = read(frozen)
        if value['case'] != item['key']: raise ValueError('Paired input belongs to another case')
        if 'dataset_arms' in config and set(value['datasets']) != set(config['dataset_arms']):
            raise ValueError('Frozen data includes a dataset excluded from this experiment')
        return value['datasets'], value['vanilla_sampled_indices']
    specific = read(item['specific_marker'])
    datasets = config.get('dataset_arms', ('vanilla_syns', 'specific_syns'))
    contexts = {}
    if 'vanilla_syns' in datasets:
        catalog = read(item['original']['source'])
        if isinstance(catalog.get('content'), dict): catalog = catalog['content']
        contexts = {row['context_id']: row['text']
                    for row in catalog.get('original_contexts', catalog.get('contexts', []))}
    result, sampling = {}, {}
    for dataset in datasets:
        result[dataset] = {}
        for n, split in enumerate(('train', 'validation')):
            target = read(specific['paths'][split])['examples']
            if dataset == 'specific_syns':
                rows = [{'question': row['question'],
                         'evidence': '\n\n'.join(chunk['text'] for chunk in
                             sorted(row['evidence']['source_chunks'], key=lambda c: c['span'][0])),
                         'source_row_index': index}
                        for index, row in enumerate(target)]
            else:
                original = read(item['original']['files'][n])['conversations']
                indices = list(range(len(original)))
                random.Random(seed+100+n).shuffle(indices)
                indices = sorted(indices[:len(target)])
                if len(indices) != len(target):
                    raise ValueError('Not enough original questions for a matched comparison')
                sampling[split] = indices
                rows = []
                for index in indices:
                    row = original[index]
                    question = row.get('question') or next(message['content']
                        for message in row['messages'] if message['role'] == 'user')
                    context_id = row.get('source_context_id') or row.get('metadata', {}).get('source_context_id')
                    evidence = contexts.get(context_id)
                    if not isinstance(evidence, str) or not evidence:
                        raise ValueError('Original question lacks its original generation context')
                    rows.append({'question': question, 'evidence': evidence, 'source_row_index': index})
            if any(not row['question'].strip() or not row['evidence'].strip() for row in rows):
                raise ValueError('Incomplete source-bound synthetic question')
            result[dataset][split] = rows
    return result, sampling


def save_conversation(conversation, path):
    from cartridges.structs import write_conversations
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+f'.{os.getpid()}.tmp.pkl')
    write_conversations([conversation], str(temp)); temp.replace(path)


def load_conversation(path, row, dataset, split):
    from cartridges.structs import read_conversations
    values = read_conversations(str(path))
    if len(values) != 1:
        raise ValueError('Teacher checkpoint must hold one conversation')
    value = values[0]
    if (value.metadata.get('original_question') != row['question']
            or value.metadata.get('dataset_arm') != dataset
            or value.metadata.get('split') != split):
        raise ValueError('Teacher checkpoint differs from the saved synthetic question')
    return value


def prepare_targets(engine, case, inputs, output, config, seed, deadline):
    import torch
    from experiments.syn_sd_factorial_runtime import FullDocumentTeacher
    from experiments.selective_learning_runtime import _conversation
    teacher = FullDocumentTeacher(engine, case['document'], config, deadline)
    targets, counts = {}, {'new': 0, 'reused': 0}
    try:
        for dataset, splits in inputs.items():
            targets[dataset] = {}
            for split, rows in splits.items():
                targets[dataset][split] = []
                for index, row in enumerate(rows):
                    if time.time() >= deadline: raise TimeoutError('Teacher target deadline')
                    path = output/'targets'/dataset/split/f'{index:03d}.pkl'
                    reused = path.is_file()
                    if reused:
                        conversation = load_conversation(path, row, dataset, split)
                        counts['reused'] += 1
                    else:
                        started = time.monotonic()
                        torch.manual_seed(seed+1000*(dataset == 'specific_syns')+100*(split == 'validation')+index)
                        example = teacher.generate(row['question'],
                            max_new_tokens=config['learning_answer_tokens'],
                            temperature=config['teacher_sampling_temperature'], topk=config['teacher_topk'])
                        conversation = _conversation(engine, row['question'], case['document'],
                                                     example, index, row['question'])
                        conversation.metadata.update(dataset_arm=dataset, split=split,
                            teacher_context='full_document', supervision='same_full_document_teacher_sparse_SD',
                            teacher_finished=example.finished, source_row_index=row['source_row_index'],
                            synthetic_answer_labels_used=False, generation_seconds=time.monotonic()-started)
                        save_conversation(conversation, path)
                        counts['new'] += 1
                        del example
                    targets[dataset][split].append(conversation)
                    print(json.dumps({'event': 'factorial_teacher_target', 'case': case['key'],
                        'dataset': dataset, 'split': split, 'index': index, 'reused': reused,
                        'answer_tokens': len(conversation.messages[-1].token_ids)}), flush=True)
    finally:
        teacher.clear(); del teacher; gc.collect(); torch.cuda.empty_cache()
    return targets, counts


def select_targets(engine, case, inputs, targets, output, config, deadline):
    import torch
    from experiments.syn_sd_factorial_runtime import SourceAssessmentGenerator
    from experiments.selective_learning_runtime import assess_teacher_signal, _legacy_helpers, SYSTEM_TEMPLATE
    helpers = _legacy_helpers(config['stack_ttt_root'])
    assessment_generator = SourceAssessmentGenerator(engine, case['document'], config, deadline)
    selected, audits = {}, {}
    try:
        for dataset, splits in inputs.items():
            selected[dataset], audits[dataset] = [], []
            for index, (row, conversation) in enumerate(zip(splits['train'], targets[dataset]['train'])):
                if time.time() >= deadline: raise TimeoutError('Selective SD preparation deadline')
                path = output/'assessments'/dataset/f'{index:03d}.json'
                candidate = conversation.messages[-1].content
                if path.is_file():
                    audit = read(path)
                    if audit['question'] != row['question'] or audit['candidate'] != candidate:
                        raise ValueError('Selective checkpoint differs from unchanged Teacher target')
                else:
                    began = time.monotonic()
                    task = SYSTEM_TEMPLATE.format(subcorpus=case['document'])+'\n\n'+row['question']
                    assessment = assess_teacher_signal(document=case['document'], question=row['question'],
                        task_prompt=task, evidence=row['evidence'], candidate=candidate,
                        candidate_complete=conversation.metadata['teacher_finished'],
                        generate_assessment=assessment_generator, legacy_helpers=helpers)
                    audit = {'question_index': index, 'question': row['question'], 'candidate': candidate,
                        'dataset': dataset, 'assessment': assessment,
                        'teacher_targets_unchanged': True, 'seconds': time.monotonic()-began,
                        'benchmark_answers_used': False, 'synthetic_answer_labels_used': False}
                    write(path, audit)
                if audit['assessment']['apply_update']:
                    selected[dataset].append(conversation)
                audits[dataset].append({'question_index': index,
                    'apply_update': audit['assessment']['apply_update'],
                    'reason': audit['assessment']['reason'], 'path': str(path)})
                print(json.dumps({'event': 'factorial_selective_signal', 'case': case['key'],
                    'dataset': dataset, 'index': index,
                    'accepted': audit['assessment']['apply_update']}), flush=True)
        transport = {'source_prefix_reused_calls': assessment_generator.cached_calls,
                     'fresh_short_context_calls': assessment_generator.fresh_calls}
    finally:
        assessment_generator.clear(); del assessment_generator; gc.collect(); torch.cuda.empty_cache()
    return selected, audits, transport


def process_case(engine, case, item, config, rank, deadline):
    import torch
    from contextualized.data import task_without_document
    from contextualized.cache_io import save_cartridge, load_cartridge
    from experiments.run_contextual_learning import train_arm
    from experiments.selective_learning_runtime import FrozenAssessmentGenerator
    from experiments.syn_sd_factorial_runtime import resident
    root = Path(config['output']); output = root/'cases'/case['key']
    seed = (config['seed']+zlib.crc32(case['key'].encode())) % (2**31)
    started = time.monotonic()
    inputs, sampling = input_rows(item, config, seed)
    manifest = {'case': case['key'], 'seed': seed, 'dataset_counts': {
        dataset: {split: len(rows) for split, rows in splits.items()} for dataset, splits in inputs.items()},
        'vanilla_sampled_source_indices': sampling, 'source': item,
        'teacher_context': config['teacher_context'], 'synthetic_answer_labels_used': False,
        'benchmark_answers_used': False}
    if (output/'input.json').is_file() and read(output/'input.json') != manifest:
        raise ValueError('Saved paired input manifest differs from this request')
    write(output/'input.json', manifest)
    baseline_path = output/'baselines.json'
    if config.get('teacher_backend') == 'vllm_subprocess':
        import subprocess, sys
        engine.student.to('cpu'); gc.collect(); torch.cuda.empty_cache()
        prepare_path=output/'teacher_prepare.log'
        prepare_log=prepare_path.open('a')
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('prepare_sd_teacher_vllm.py')),
            '--config', config['_config_path'], '--case', case['key']], start_new_session=True,
            stdout=prepare_log,stderr=subprocess.STDOUT)
        try:
            status = child.wait(timeout=max(1, deadline-time.time()-5))
            if status:
                raise TeacherPreparationError(f'Teacher preparation exited with status {status}; log: {prepare_path}')
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, 9); child.wait(); raise TimeoutError('Teacher preparation deadline')
        finally:
            try: os.killpg(child.pid, 15)
            except ProcessLookupError: pass
            prepare_log.close()
        baselines = read(baseline_path); targets, selected = {}, {}
        supervision = read(output/'supervision_summary.json')
        for dataset, splits in inputs.items():
            targets[dataset] = {split: [load_conversation(output/'targets'/dataset/split/f'{index:03d}.pkl',
                row, dataset, split) for index, row in enumerate(rows)] for split, rows in splits.items()}
            admitted = {entry['question_index'] for entry in supervision['gate_audits'][dataset] if entry['apply_update']}
            selected[dataset] = [value for index,value in enumerate(targets[dataset]['train']) if index in admitted]
        target_counts = supervision['target_cache']; gate_audits = supervision['gate_audits']; transport = supervision['transport']
    else:
        resident(engine, 'teacher')
        if baseline_path.is_file():
            baselines = read(baseline_path)
        else:
            generate = FrozenAssessmentGenerator(engine, prefill_tokens=config['prefill_tokens'])
            full = generate(case['prompt'], config['eval_tokens'])
            bare = generate(task_without_document(case), config['eval_tokens'])
            baselines = {'full_context_teacher': full['prediction'], 'no_context': bare['prediction'],
                'full_teacher_prompt': 'original_benchmark_task_and_full_document',
                'full_teacher_complete': full['complete'], 'no_context_complete': bare['complete']}
            write(baseline_path, baselines)
        targets, target_counts = prepare_targets(engine, case, inputs, output, config, seed, deadline)
        selected, gate_audits, transport = select_targets(engine, case, inputs, targets, output, config, deadline)
    write(output/'supervision_summary.json', {'target_cache': target_counts,
        'train_counts': {dataset: {'vanilla_sd': len(targets[dataset]['train']),
                                  'selective_sd': len(selected[dataset])} for dataset in inputs},
        'gate': config['selective_gate'], 'gate_audits': gate_audits, 'transport': transport,
        'heldout_policy': config['validation_policy'], 'teacher_targets_unchanged': True})
    resident(engine, 'student')
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    initial_path = output/'initial.safetensors'
    if initial_path.is_file(): initial, _ = load_cartridge(initial_path, device=engine.device)
    else:
        initial = engine.initial_cache(case['document'])
        save_cartridge(initial, initial_path, config['model'])
    query = task_without_document(case)
    if 'initial_kv' not in baselines:
        baselines['initial_kv'] = engine.generate_student(query, initial, max_new_tokens=config['eval_tokens'])[0]
        initial.clear(); write(baseline_path, baselines)
    results = {}
    for dataset in config.get('dataset_arms', ('vanilla_syns', 'specific_syns')):
        for sd in config.get('sd_arms', ('oracle_sd', 'selective_sd')):
            name = dataset+'_'+sd; arm_path = output/'arms'/(name+'.json')
            if arm_path.is_file():
                results[name] = read(arm_path)
                continue
            if time.time() >= deadline: raise TimeoutError('Native KV adaptation deadline')
            values = targets[dataset]['train'] if sd in {'oracle_sd','vanilla_sd'} else selected[dataset]
            fallback=False
            if not values and config.get('empty_selection_policy')=='original_SD':
                values=targets[dataset]['train'];fallback=True
            began = time.monotonic()
            arm_config=dict(config)
            reference=config.get('matched_update_reference_arm')
            if reference and reference!=name:
                arm_config['fixed_updates']=results[reference]['training']['updates']
            cache, training = train_arm(engine, initial, values, targets[dataset]['validation'], arm_config, seed)
            training['all_rejected_original_SD_fallback']=fallback
            prediction, tokens = engine.generate_student(query, cache, max_new_tokens=config['eval_tokens'])
            cache.clear()
            delta_sq = 0.
            for trained, base in zip(cache.parameters(), initial.parameters()):
                if trained.requires_grad:
                    delta_sq += float((trained.detach().float()-base.detach().float()).square().sum())
            cartridge = output/'cartridges'/(name+'.safetensors')
            save_cartridge(cache, cartridge, config['model'])
            result = {'case': case['key'], 'dataset': dataset, 'sd': sd, 'status': 'completed',
                'prediction': prediction, 'generated_token_ids': list(map(int, tokens)),
                'training': training, 'native_kv_parameter_change_l2': delta_sq**.5,
                'native_kv_only': True, 'backbone_frozen': True,
                'inference_reads_original_document': False, 'cartridge': str(cartridge),
                'seconds': time.monotonic()-began, 'benchmark_answers_used': False,
                'synthetic_answer_labels_used': False,
                'no_accepted_signal': sd=='selective_sd' and not bool(selected[dataset]),
                'answer_always_generated': True}
            write(arm_path, result); results[name] = result
            print(json.dumps({'event': 'factorial_arm_complete', 'case': case['key'], 'arm': name,
                'updates': training['updates'], 'seconds': result['seconds']}), flush=True)
            del cache; gc.collect(); torch.cuda.empty_cache()
    initial.clear(); del initial
    record = {'case': case['key'], 'query': case['query'], 'task': item['task'], 'length': item['length'],
        'status': 'completed', 'protocol': config['protocol'], 'rank': rank,
        'predictions': {name: arm['prediction'] for name, arm in results.items()},
        'baselines': baselines, 'arms': results, 'inputs': manifest,
        'supervision_summary': str(output/'supervision_summary.json'),
        'seconds': time.monotonic()-started, 'completed_epoch': time.time(),
        'benchmark_answers_used_for_training': False, 'native_kv_only': True,
        'analysis_split':item.get('analysis_split','unspecified'),
        'gpu': torch.cuda.get_device_name()}
    write(root/'results/cases'/(case['key']+'.json'), record)
    return record


def run(config, rank):
    from contextualized.data import read_indexed_case
    from experiments.syn_sd_factorial_runtime import load_engine
    root = Path(config['output']); source = Path(config['dataset_root'])
    readiness = read(source/'synthetic_dataset_export_progress.json')
    if not readiness.get('complete') or readiness['counts']['completed_data_cases'] != 1000:
        raise ValueError('The 1000-case synthetic dataset must finish before SD starts')
    if not 0 <= rank < config['workers'] <= 12: raise ValueError('Invalid GPU worker rank')
    deadline = min(config['deadline_epoch'], float(os.environ.get('PAIR_DEADLINE_EPOCH', config['deadline_epoch'])))
    plan = read(config['source_plan']); index = {row['key']: row for row in read(plan['source_index'])}
    engine = None; done, failed = [], []
    for item in config['cases']:
        key = item['key']; result = root/'results/cases'/(key+'.json')
        if result.is_file(): continue
        if time.time() >= deadline-60: break
        if item.get('analysis_split')=='holdout' and config.get('phase_control_path'):
            while read(config['phase_control_path'])['phase']=='development' and time.time()<deadline-60:
                time.sleep(5)
            if time.time()>=deadline-60 or read(config['phase_control_path'])['phase']=='stop':break
        lock_path = root/'claims'/(key+'.lock'); lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open('a+')
        try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close(); continue
        try:
            if result.is_file() or (root/'results/failures'/(key+'.json')).is_file(): continue
            if engine is None: engine = load_engine(config)
            case = {**read_indexed_case(plan, index, key), 'task': item['task'], 'length': item['length']}
            print(json.dumps({'event': 'factorial_case_started', 'case': key, 'rank': rank}), flush=True)
            process_case(engine, case, item, config, rank, deadline); done.append(key)
        except TimeoutError as error:
            write(root/'results/partial'/(key+'.json'), {'case': key, 'error': str(error),
                'status': 'deadline_checkpoint', 'rank': rank, 'epoch': time.time()})
            break
        except TeacherPreparationError as error:
            failed.append(key)
            write(root/'results/partial'/(key+'.json'),{'case':key,'error':str(error),
                'status':'retryable_teacher_process_failure','rank':rank,'epoch':time.time(),
                'case_not_excluded_from_other_workers':True})
            print(traceback.format_exc(),flush=True)
            break
        except Exception as error:
            failed.append(key)
            write(root/'results/failures'/(key+'.json'), {'case': key, 'rank': rank,
                'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(), 'epoch': time.time()})
            print(traceback.format_exc(), flush=True)
            import torch
            if isinstance(error, torch.OutOfMemoryError) or 'CUDA' in str(error): break
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN); stream.close()
        write(root/'results/workers'/f'{rank:02d}.json', {'rank': rank, 'completed': done,
            'failed': failed, 'epoch': time.time(), 'terminal': False})
    write(root/'results/workers'/f'{rank:02d}.json', {'rank': rank, 'completed': done,
        'failed': failed, 'epoch': time.time(), 'terminal': True})
    return 1 if failed else 75 if time.time() >= deadline else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--rank', required=True, type=int)
    a = p.parse_args(); config = read(a.config); config['_config_path'] = a.config
    return run(config, a.rank)


if __name__ == '__main__': raise SystemExit(main())
