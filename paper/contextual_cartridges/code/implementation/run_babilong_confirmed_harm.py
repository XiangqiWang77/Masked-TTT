"""Frozen BABILong 2x2 data/SD comparison using matched confirmed-harm refits.

This worker never loads benchmark answers. Official scoring runs in a separate
reporting process after predictions have been saved; it cannot select a model.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import zlib

from experiments.run_syn_sd_factorial import (input_rows, load_conversation,
    materialize_training_prompts, prepare_targets, read, write)


DATASETS = ('vanilla_syns', 'specific_syns')
MATRIX = tuple(f'{dataset}_{sd}' for dataset in DATASETS
               for sd in ('vanilla_sd', 'selective_sd'))
PROTOCOL = 'babilong-confirmed-harm-tracin-v1'


def matched_config(base, *, reference_updates=None):
    value = dict(base)
    if reference_updates is not None:
        if reference_updates < 1:
            raise ValueError('The query-agnostic pilot must have positive updates')
        value.pop('official_epochs', None)
        value.update(fixed_updates=reference_updates, train_batch_size=1,
                     min_steps=1, max_steps=reference_updates)
    return value


def expected_answer_ids(engine, row):
    if row.get('answer_token_ids') is not None:
        return list(map(int, row['answer_token_ids']))
    tokens = engine.tokenizer.encode(row['answer'], add_special_tokens=False)
    if tokens and row.get('answer_finished', True) and tokens[-1] not in engine.eos_ids:
        tokens.append(int(engine.tokenizer.eos_token_id))
    return tokens


def recover_teacher_targets(engine, inputs, output, config, key):
    """Reuse only matching source, question, fixed answer and actual soft targets."""
    from experiments.selective_learning_runtime import SYSTEM_TEMPLATE
    reused = 0
    for dataset, splits in inputs.items():
        for split, rows in splits.items():
            for index, row in enumerate(rows):
                relative = Path('targets') / dataset / split / f'{index:03d}.pkl'
                destination = output / relative
                if destination.exists():
                    continue
                for root in config.get('teacher_cache_sources', []):
                    source = Path(root) / 'cases' / key / relative
                    if not source.exists():
                        continue
                    try:
                        conversation = load_conversation(source, row, dataset, split)
                        valid = (
                            conversation.system_prompt == SYSTEM_TEMPLATE.format(subcorpus=row['evidence'])
                            and conversation.metadata.get('synthetic_answer_trajectory_used') is True
                            and list(conversation.messages[-1].token_ids) == expected_answer_ids(engine, row)
                            and conversation.messages[-1].top_logprobs is not None)
                    except (ValueError, KeyError):
                        valid = False
                    if valid:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.link(source, destination)
                        reused += 1
                        break
    return reused


def parameter_delta(cache, initial):
    import torch
    with torch.no_grad():
        return sum(float((p.detach().float() - q.detach().float()).square().sum())
                   for p, q in zip(cache.parameters(), initial.parameters())
                   if p.requires_grad) ** .5


def process_case(engine, case, item, config, deadline):
    import torch
    from contextualized.cache_io import load_cartridge, save_cartridge
    from contextualized.data import task_without_document
    from experiments.conservative_tracin_runtime import train_conservatively
    from experiments.run_figure4_contextual import complete_target_layout, training_config
    from experiments.syn_sd_factorial_runtime import resident

    root = Path(config['output'])
    output = root / 'cases' / case['key']
    seed = (config['seed'] + zlib.crc32(case['key'].encode())) % (2**31)
    started = time.monotonic()
    inputs, sampling = input_rows(item, config, seed)
    materialize_training_prompts(case, inputs, config)
    # Keep existing train split exactly; validation data is not a loss,
    # selector reference, extra training set or Teacher-generation workload.
    inputs = {name: {'train': inputs[name]['train'], 'validation': []} for name in DATASETS}
    if any(not value['train'] for value in inputs.values()):
        raise ValueError('Both frozen datasets must contain training trajectories')
    manifest = {'case': case['key'], 'seed': seed, 'datasets': inputs,
                'vanilla_sampled_indices': sampling, 'benchmark_answers_used': False}
    path = output / 'input.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Frozen training inputs changed on resume')
    write(path, manifest)
    recovered = recover_teacher_targets(engine, inputs, output, config, case['key'])
    missing = any(not (output / 'targets' / name / 'train' / f'{index:03d}.pkl').exists()
                  for name in DATASETS for index in range(len(inputs[name]['train'])))
    if missing:
        resident(engine, 'teacher')
    targets, target_counts = prepare_targets(engine, case, inputs, output, config, seed, deadline)
    resident(engine, 'student')
    base = training_config(config['model'], config['prefix_tokens'])
    base.update(protocol=PROTOCOL, packed_tokens=config['packed_tokens'],
                learning_rate=config['learning_rate'], official_epochs=config['official_epochs'],
                matched_update_reference_arm='vanilla_syns_vanilla_sd')
    base, packing = complete_target_layout(engine,
        [conversation for name in DATASETS for conversation in targets[name]['train']], base)
    engine.dataset_packing_mode = base['packing_mode']
    engine.packed_tokens = base['packed_tokens']
    engine.sd_target_only_logits = True
    engine.sd_offload_saved_tensors = True
    initial_path = output / 'initial.safetensors'
    if initial_path.exists():
        initial, _ = load_cartridge(initial_path, device=engine.device,
            expected_prefix_tokens=config['prefix_tokens'], expected_model=config['model'])
    else:
        random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        initial = engine.initial_cache(case['document'])
        save_cartridge(initial, initial_path, config['model'])
    query = task_without_document(case)
    reference_updates, results = None, {}
    for name in DATASETS:
        names = [f'{name}_vanilla_sd', f'{name}_selective_sd']
        paths = [output / 'arms' / f'{arm}.json' for arm in names]
        if all(path.exists() for path in paths):
            for arm, path in zip(names, paths):
                results[arm] = read(path)
            if name == DATASETS[0]:
                reference_updates = results[names[0]]['training']['updates']
            continue
        if time.time() >= deadline - 60:
            raise TimeoutError('Development allocation deadline')
        study_config = matched_config(base, reference_updates=reference_updates)
        digest = hashlib.sha256()
        for path in sorted((output / 'targets' / name / 'train').glob('*.pkl')):
            digest.update(path.read_bytes())
        study_config['teacher_targets_sha256'] = digest.hexdigest()
        study = output / 'work' / PROTOCOL / name
        began = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        cache, selection = train_conservatively(engine, initial, targets[name]['train'],
            study_config, seed, study, deadline=deadline)
        readout_started = time.monotonic()
        prediction, token_ids = engine.generate_student(query, cache,
                                                        max_new_tokens=config['eval_tokens'])
        selective_readout_seconds = time.monotonic() - readout_started
        delta = parameter_delta(cache, initial)
        save_cartridge(cache, study / 'contextual.safetensors', config['model'])
        cache.clear(); del cache
        gc.collect(); torch.cuda.empty_cache()
        if selection['removed']:
            readout_started = time.monotonic()
            pilot, _ = load_cartridge(selection['vanilla_checkpoint'], device=engine.device,
                expected_prefix_tokens=config['prefix_tokens'], expected_model=config['model'])
            pilot_prediction, pilot_ids = engine.generate_student(query, pilot,
                max_new_tokens=config['eval_tokens'])
            pilot_delta = parameter_delta(pilot, initial)
            pilot.clear(); del pilot
            pilot_readout_seconds = time.monotonic() - readout_started
        else:
            pilot_prediction, pilot_ids, pilot_delta = prediction, token_ids, delta
            pilot_readout_seconds = selective_readout_seconds
        if reference_updates is None:
            reference_updates = selection['vanilla_training']['updates']
        study_seconds = time.monotonic() - began
        for arm, sd, text, ids, change, training, checkpoint in [
            (names[0], 'vanilla_sd', pilot_prediction, pilot_ids, pilot_delta,
             selection['vanilla_training'], selection['vanilla_checkpoint']),
            (names[1], 'selective_sd', prediction, token_ids, delta,
             selection['selective_training'], str(study / 'contextual.safetensors'))]:
            result = {'case': case['key'], 'dataset': name, 'sd': sd, 'status': 'completed',
                'protocol': PROTOCOL, 'prediction': text, 'generated_token_ids': list(map(int, ids)),
                'training': training, 'native_kv_parameter_change_l2': change,
                'cartridge': checkpoint,
                'seconds': (selection['tracin']['original_fits'][0]['seconds'] + pilot_readout_seconds
                            if sd == 'vanilla_sd' else
                            study_seconds - (pilot_readout_seconds if selection['removed'] else 0)),
                'timing_scope': 'training_and_readout_excluding_shared_teacher_preparation',
                'shared_dataset_execution_seconds': study_seconds,
                'no_accepted_signal': False,
                'no_confirmed_deletion': sd == 'selective_sd' and selection['removed'] == 0,
                'native_kv_only': True, 'backbone_frozen': True,
                'benchmark_answers_used': False, 'synthetic_answer_hard_label_loss': False,
                'inference_reads_original_document': False,
                'vanilla_pilot_reused_by_selector': True,
                'training_packing': packing,
                'peak_student_memory_allocated_bytes': torch.cuda.max_memory_allocated()}
            if sd == 'selective_sd':
                result['confirmed_harm_selection'] = selection
            write(output / 'arms' / f'{arm}.json', result)
            results[arm] = result
        # Keep the actual ordinary pilot and final memory; retain every refit
        # decision/loss in JSON without accumulating all temporary KV tensors.
        for checkpoint in (study / 'fits').glob('*/checkpoint.safetensors'):
            if str(checkpoint) != selection['vanilla_checkpoint']:
                checkpoint.unlink()
        gc.collect(); torch.cuda.empty_cache()
    initial.clear(); del initial
    record = {'case': case['key'], 'query': case['query'], 'task': item['task'],
        'length': item['length'], 'status': 'completed', 'protocol': PROTOCOL,
        'predictions': {arm: value['prediction'] for arm, value in results.items()},
        'arms': results, 'baselines': {}, 'seconds': time.monotonic() - started,
        'completed_epoch': time.time(), 'benchmark_answers_used_for_training': False,
        'target_cache': {**target_counts, 'recovered_prior_targets': recovered},
        'new_synthetic_generation': False, 'analysis_split': item.get('analysis_split')}
    write(root / 'results/cases' / f"{case['key']}.json", record)
    print(json.dumps({'event': 'babilong_2x2_complete', 'case': case['key'],
                      'seconds': record['seconds'], 'arms': list(results)}), flush=True)


def run(config, rank):
    from contextualized.data import read_indexed_case
    from experiments.syn_sd_factorial_runtime import load_engine
    if rank != 0 or config['workers'] != 1 or tuple(config['matrix']) != MATRIX:
        raise ValueError('This development experiment uses one GPU for the complete 2x2 matrix')
    root = Path(config['output'])
    deadline = min(config['deadline_epoch'], float(os.environ['PAIR_DEADLINE_EPOCH']))
    plan = read(config['source_plan'])
    index = {row['key']: row for row in read(plan['source_index'])}
    engine = None
    for item in config['cases']:
        key = item['key']
        if (root / 'results/cases' / f'{key}.json').exists():
            continue
        if time.time() >= deadline - 180:
            return 75
        try:
            if engine is None:
                engine = load_engine(config)
            case = {**read_indexed_case(plan, index, key),
                    'task': item['task'], 'length': item['length']}
            process_case(engine, case, item, config, deadline)
            with (root / 'report.log').open('a') as log:
                subprocess.run([sys.executable, '-m', 'experiments.report_syn_sd_factorial',
                    '--config', config['_config_path']], stdout=log, stderr=log, check=True)
        except TimeoutError as error:
            write(root / 'results/partial' / f'{key}.json',
                  {'case': key, 'error': str(error), 'epoch': time.time()})
            return 75
        except Exception:
            write(root / 'results/partial' / f'{key}.json',
                  {'case': key, 'traceback': traceback.format_exc(), 'epoch': time.time(),
                   'case_not_excluded': True})
            raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--rank', required=True, type=int)
    args = parser.parse_args()
    config = read(args.config)
    config['_config_path'] = args.config
    return run(config, args.rank)


if __name__ == '__main__':
    raise SystemExit(main())
