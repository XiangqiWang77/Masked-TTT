"""Preserve a development run while repairing post-update checkpoint selection."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

from experiments.run_syn_sd_factorial import read, write


def prepare(config_path, destination):
    old = read(config_path)
    source = Path(old['output'])
    root = Path(destination)
    if (root/'config.json').exists():
        raise FileExistsError('Do not overwrite an existing experiment')
    config = deepcopy(old)
    config.update(output=str(root),
        protocol='native-Cartridges-source-teacher-question-guided-grounded-veto-earlybest-v4',
        track_best_before_min=True,
        runtime_cache_root=str(source/'runtime_caches'),
        phase_control_path=str(root/'phase.json'),
        protocol_frozen_utc=datetime.now(timezone.utc).isoformat(),
        previous_checkpoint_policy_run=str(source),
        checkpoint_policy='best heldout checkpoint after a positive number of updates; minimum executed updates unchanged')
    reused = Counter()
    for item in config['cases']:
        previous = source/'cases'/item['key']
        target = root/'cases'/item['key']
        for name in ('initial.safetensors', 'baselines.json'):
            path = previous/name
            if path.is_file():
                (target/name).parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(path, target/name)
                except OSError:
                    shutil.copy2(path, target/name)
                reused[name] += 1
        for directory in ('targets', 'assessments'):
            for path in (previous/directory).rglob('*'):
                if path.is_file() and path.stem.isdigit() and path.suffix in ('.pkl', '.json'):
                    dest = target/path.relative_to(previous)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(path, dest)
                    except OSError:
                        shutil.copy2(path, dest)
                    reused[directory] += 1
    # Only immutable Teacher/gate inputs and initialization are reused. Every
    # fitted memory and answer is recomputed under the repaired common policy.
    for name in ('success_criteria.json', 'evaluation_partitions.json'):
        if (source/name).exists():
            shutil.copy2(source/name, root/name)
    write(root/'phase.json', {'phase': 'development', 'holdout_enabled': False,
        'reason': 'universal post-update checkpoint repair; evaluate on development before releasing holdout'})
    write(root/'protocol_audit.json', {
        'previous_run': str(source), 'previous_protocol': old['protocol'],
        'change': 'retain best positive-update validation checkpoint before min_steps while preserving minimum executed updates',
        'motivation': 'development KD histories showed earlier improved checkpoints discarded by the min-step eligibility rule',
        'all_four_arms_use_same_checkpoint_policy': True,
        'same_actual_update_budget': True,
        'teacher_prompts_sampling_soft_targets_and_admission_unchanged': True,
        'synthetic_datasets_regenerated': False,
        'earlier_trained_memories_or_answers_reused': False,
        'previous_results_preserved': True,
        'benchmark_answers_not_used_to_select_checkpoint': True,
        'original_deadline_unchanged': config['deadline_utc'],
        'reused': dict(reused)})
    write(root/'config.json', config)
    return config, reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config, reused = prepare(args.previous_config, args.output)
    print({'config': str(Path(config['output'])/'config.json'), 'reused': dict(reused),
           'minimum_executed_updates': config['min_steps'],
           'track_best_before_min': config['track_best_before_min'],
           'deadline_utc': config['deadline_utc']})


if __name__ == '__main__':
    main()
