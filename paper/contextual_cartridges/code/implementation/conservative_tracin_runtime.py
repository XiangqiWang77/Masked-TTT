"""Disk-backed matched refits for conservative native-KV selective SD.

Only one trial Cartridge/optimizer is on the GPU at a time. Teacher targets,
the initial KV, learning rate and final update budget are fixed across trials.
No evaluation answer or external validation data is accepted by this API.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

from contextualized.conservative_tracin import select_by_confirmed_harm


class SelectiveDeadline(TimeoutError):
    """Save completed fits and stop without labeling an unfinished case complete."""


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def key_for(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


class MatchedRefits:
    def __init__(self, engine, initial, conversations, config, seed, directory, deadline):
        self.engine, self.initial = engine, initial
        self.conversations, self.config = conversations, config
        self.seed, self.directory, self.deadline = seed, Path(directory), deadline
        self.update_budget = None
        self.fit_records = {}
        self.actual_fits = self.reused_fits = 0

    def check(self):
        if time.time() >= self.deadline:
            raise SelectiveDeadline('12-hour allocation budget reached; completed fits are saved')

    def event(self, **event):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / 'compute_events.jsonl').open('a') as handle:
            handle.write(json.dumps({'epoch': time.time(), **event}, allow_nan=False) + '\n')

    def fit(self, indices, seed):
        import gc
        import torch
        from contextualized.cache_io import load_cartridge, save_cartridge
        from contextualized.early_study import evaluate_distillation, make_dataset
        from experiments.run_contextual_learning import train_arm

        self.check()
        key = key_for({'indices': indices, 'seed': seed})
        directory = self.directory / 'fits' / key
        checkpoint = directory / 'checkpoint.safetensors'
        metadata = directory / 'fit.json'
        if checkpoint.exists() and metadata.exists():
            record = read(metadata)
            if record['indices'] != list(indices) or record['seed'] != seed:
                raise ValueError('Cached trial identity changed')
            checked, _ = load_cartridge(checkpoint, device='cpu',
                expected_prefix_tokens=self.initial.num_cartridge_tokens(),
                expected_model=self.config['model'])
            del checked
            self.reused_fits += 1
        else:
            config = dict(self.config)
            if self.update_budget is not None:
                config.pop('official_epochs', None)
                config.update(fixed_updates=self.update_budget, train_batch_size=1,
                              min_steps=1, max_steps=self.update_budget)
            started = time.monotonic()
            self.event(event='fit_start', key=key, indices=indices, seed=seed,
                       fixed_updates=self.update_budget)
            def progress(row):
                self.event(event='optimizer_step', key=key, step=row['step'])
                self.check()
            torch.manual_seed(seed)
            if torch.device(self.engine.device).type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            cache = None
            try:
                cache, training = train_arm(self.engine, self.initial,
                    [self.conversations[index] for index in indices], [], config, seed,
                    progress=progress)
                if training['updates'] < 1:
                    raise ValueError('A matched fit must update native KV')
                losses = []
                for index, conversation in enumerate(self.conversations):
                    self.check()
                    data = make_dataset(self.engine, [conversation],
                                        self.config['packed_tokens'], self.seed + index)
                    loss = evaluate_distillation(self.engine, cache, data)
                    if not math.isfinite(loss):
                        raise FloatingPointError('Nonfinite per-trajectory SD loss')
                    losses.append(loss)
                save_cartridge(cache, checkpoint, self.config['model'])
                record = {'key': key, 'checkpoint': str(checkpoint), 'indices': list(indices),
                          'seed': seed, 'updates': training['updates'], 'losses': losses,
                          'training': training, 'seconds': time.monotonic() - started,
                          'all_losses_are_training_SD': True, 'benchmark_answers_used': False}
                write(metadata, record)
                self.event(event='fit_complete', key=key, updates=record['updates'],
                           seconds=record['seconds'])
                self.actual_fits += 1
            finally:
                if cache is not None:
                    cache.clear()
                    del cache
                gc.collect()
                if torch.device(self.engine.device).type == 'cuda':
                    torch.cuda.empty_cache()
        if self.update_budget is None:
            if list(indices) != list(range(len(self.conversations))) or seed != self.seed:
                raise ValueError('The primary full-data pilot must set the update budget')
            self.update_budget = record['updates']
        if record['updates'] != self.update_budget:
            raise ValueError('Cached or new trial did not use the matched update budget')
        self.fit_records[key] = record
        return record

    def score(self, indices, fitted):
        import gc
        import torch
        from contextualized.cache_io import load_cartridge
        from contextualized.tracin_sd import score_checkpoint

        self.check()
        signature = {'indices': indices, 'fits': [row['key'] for row in fitted]}
        destination = self.directory / 'scores' / (key_for(signature) + '.json')
        if destination.exists():
            saved = read(destination)
            if saved['signature'] != signature:
                raise ValueError('Saved TracIn checkpoints do not match the trial references')
            return saved['rows']
        rows, audits = [], []
        active = [self.conversations[index] for index in indices]
        # Every seed must agree at its final state as well as at shared Z0.
        for index, fitted_record in enumerate([None, *fitted]):
            self.check()
            started = time.monotonic()
            self.event(event='checkpoint_score_start', checkpoint_index=index,
                       active_indices=indices)
            cache = self.initial
            try:
                if fitted_record is not None:
                    cache, _ = load_cartridge(fitted_record['checkpoint'],
                        device=self.engine.device,
                        expected_prefix_tokens=self.initial.num_cartridge_tokens(),
                        expected_model=self.config['model'])
                values, audit = score_checkpoint(self.engine, cache, active, active,
                    packed_tokens=self.config['packed_tokens'], seed=self.seed,
                    learning_rate=self.config['learning_rate'], check=self.check,
                    stream_reference=True)
                rows.append(values)
                audits.append(audit)
            finally:
                cache.clear()
                if fitted_record is not None:
                    del cache
                gc.collect()
                if torch.device(self.engine.device).type == 'cuda':
                    torch.cuda.empty_cache()
            self.event(event='checkpoint_score_complete', checkpoint_index=index,
                       seconds=time.monotonic() - started)
        write(destination, {'signature': signature, 'rows': rows, 'audits': audits,
                            'initial_magnitude_cannot_override_other_checkpoints': True})
        return rows


def train_conservatively(engine, initial, conversations, config, seed, directory, *, deadline):
    from contextualized.cache_io import load_cartridge
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = {'config': config, 'seed': seed,
                'actual_prefix_tokens': initial.num_cartridge_tokens()}
    path = directory / 'training_contract.json'
    if path.exists() and read(path) != contract:
        raise ValueError('Saved training checkpoint uses different cache/data/training settings')
    if not conversations:
        raise ValueError('Conservative SD requires usable Teacher trajectories')
    write(path, contract)
    store = MatchedRefits(engine, initial, conversations, config, seed, directory, deadline)
    seeds = [seed, seed + 100003]
    result = select_by_confirmed_harm(len(conversations), seeds=seeds,
        fit=store.fit, score=store.score,
        checkpoint=lambda state: write(directory / 'selection_progress.json', state),
        atol=config['deletion_loss_atol'], rtol=config['deletion_loss_rtol'])
    primary = result['primary_fit']
    original = result['original_fits'][0]
    result['compute'] = {
        'new_fits_this_attempt': store.actual_fits, 'reused_fits_this_attempt': store.reused_fits,
        'unique_completed_fits_used': len(store.fit_records),
        'optimizer_updates_in_completed_fits_used': sum(row['updates'] for row in store.fit_records.values()),
        'returned_checkpoint_updates': primary['updates'],
        'extra_selection_compute_included': True,
        'event_log_includes_interrupted_attempts': str(directory / 'compute_events.jsonl'),
    }
    write(directory / 'selection.json', result)
    cache, _ = load_cartridge(primary['checkpoint'], device=engine.device,
        expected_prefix_tokens=initial.num_cartridge_tokens(), expected_model=config['model'])
    return cache, {
        'fallback': result['fallback'], 'kept': len(result['kept_indices']),
        'removed': len(result['removed_indices']), 'kept_indices': result['kept_indices'],
        'removed_indices': result['removed_indices'], 'tracin': result,
        'vanilla_training': original['training'], 'selective_training': primary['training'],
        'vanilla_checkpoint': original['checkpoint'],
        'matched_update_reference': 'same_case_query_specific_vanilla_pilot',
        'source_gate_auto_deletions': 0,
        'positive_zero_and_uncertain_signals_kept_by_default': True,
        'accuracy_nonregression_guaranteed': False,
        'trial_checkpoints_temporary_until_case_completion': True,
    }
