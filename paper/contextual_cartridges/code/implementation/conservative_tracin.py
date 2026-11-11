"""Conservative TracIn selection with measured, per-trajectory deletion tests.

This module does not judge factual correctness.  It nominates a trajectory only
when every checkpoint calls it an opponent, then requires actual matched-budget
refits to improve the remaining SD objective without worsening any survivor.
The callbacks contain the ordinary native-KV trainer; no gold/heldout labels are
inputs to this selector.
"""
from __future__ import annotations

import math


def unanimous_opponents(checkpoints):
    """Do not sum checkpoints: a large initial gradient cannot overrule a veto."""
    if len(checkpoints) < 2:
        raise ValueError('Consistent harm requires at least two checkpoints')
    width = len(checkpoints[0])
    if any(len(rows) != width for rows in checkpoints):
        raise ValueError('TracIn checkpoints have different trajectory sets')
    result = []
    for index in range(width):
        rows = [checkpoint[index] for checkpoint in checkpoints]
        if any(row['candidate_index'] != index for row in rows):
            raise ValueError('TracIn checkpoint trajectory order changed')
        dots = [float(row['gradient_dot']) for row in rows]
        cosines = [float(row['gradient_cosine']) for row in rows]
        if not all(math.isfinite(value) for value in dots + cosines):
            raise FloatingPointError('Nonfinite TracIn evidence')
        insufficient = any(row.get('insufficient_other_reference_terms', False)
                           for row in rows)
        nominate = not insufficient and all(value < 0 for value in dots)
        result.append({
            'candidate_index': index, 'nominate_removal': nominate,
            'checkpoint_dots': dots, 'checkpoint_cosines': cosines,
            'least_negative_cosine': max(cosines),
            'reason': ('negative_at_every_checkpoint' if nominate else
                       'insufficient_other_trajectories' if insufficient else
                       'nonnegative_or_disagreeing_checkpoints_keep'),
        })
    return result


def deletion_verdict(current, trial, original, survivors, *, atol=1e-6, rtol=1e-5):
    """Require agreement across training orders and protect EACH surviving loss.

    Original-pilot comparisons prevent tolerance drift across several accepted
    deletions. Improvement is measured only on survivors; rejected trajectories
    do not get a vote on subsequent consolidation.
    """
    if len(current) < 2 or not (len(current) == len(trial) == len(original)):
        raise ValueError('Deletion tests require at least two matched training orders')
    if not survivors or len(set(survivors)) != len(survivors):
        raise ValueError('A deletion must leave distinct protected trajectories')
    if atol < 0 or rtol < 0:
        raise ValueError('Numerical tolerances cannot be negative')
    checks = []
    for before, after, baseline in zip(current, trial, original):
        if not (before['updates'] == after['updates'] == baseline['updates'] > 0):
            raise ValueError('A deletion trial changed the optimizer-update budget')
        if not (before['seed'] == after['seed'] == baseline['seed']):
            raise ValueError('A deletion trial changed its matched training seed')
        losses = [record['losses'] for record in (before, after, baseline)]
        if len({len(values) for values in losses}) != 1:
            raise ValueError('Loss vectors have different trajectory identities')
        if not all(math.isfinite(float(value)) for values in losses for value in values):
            raise FloatingPointError('Nonfinite SD loss in a deletion trial')
        deltas, baseline_deltas, allowed, baseline_allowed = [], [], [], []
        for index in survivors:
            old, new, base = (float(values[index]) for values in losses)
            deltas.append(new - old)
            baseline_deltas.append(new - base)
            allowed.append(atol + rtol * abs(old))
            baseline_allowed.append(atol + rtol * abs(base))
        no_current_regression = all(delta <= tolerance
                                    for delta, tolerance in zip(deltas, allowed))
        no_original_regression = all(delta <= tolerance for delta, tolerance in
                                     zip(baseline_deltas, baseline_allowed))
        mean_improvement = -sum(deltas) / len(deltas)
        strict_improvement = mean_improvement > sum(allowed) / len(allowed)
        checks.append({
            'seed': before['seed'], 'survivor_indices': list(survivors),
            'per_trajectory_loss_change': deltas,
            'per_trajectory_change_from_original': baseline_deltas,
            'per_trajectory_tolerance': allowed,
            'per_trajectory_original_tolerance': baseline_allowed,
            'mean_loss_improvement': mean_improvement,
            'no_current_regression': no_current_regression,
            'no_original_regression': no_original_regression,
            'strict_mean_improvement': strict_improvement,
            'passed': no_current_regression and no_original_regression and strict_improvement,
        })
    return {
        'accept_removal': all(check['passed'] for check in checks),
        'checks': checks, 'absolute_tolerance': atol, 'relative_tolerance': rtol,
        'all_survivors_and_all_training_orders_required': True,
        'factual_correctness_guarantee': False,
    }


def select_by_confirmed_harm(count, *, seeds, fit, score, checkpoint=None,
                             atol=1e-6, rtol=1e-5):
    """Greedy deletion with fresh TracIn evidence after every accepted change.

    fit(indices, seed) returns a disk-backed record with one loss per ORIGINAL
    trajectory. score(indices, fitted_records) returns checkpoint-wise LOO
    gradient scores in the supplied index order. All trials start at the same
    Z0 with fresh optimizer state; fit enforces a common positive update budget.
    """
    if count < 1 or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Need nonempty data and distinct matched training orders')
    active = list(range(count))
    original = [fit(active, seed) for seed in seeds]
    current = original
    rounds, removed = [], []
    while len(active) > 1:
        contributions = score(active, current)
        nominations = unanimous_opponents(contributions)
        if len(nominations) != len(active):
            raise ValueError('Scored trajectory set differs from the retained set')
        candidates = sorted((row for row in nominations if row['nominate_removal']),
                            key=lambda row: (row['least_negative_cosine'], row['candidate_index']))
        round_log = {'active_indices': list(active), 'nominations': nominations, 'trials': []}
        accepted = False
        for candidate in candidates:
            index = active[candidate['candidate_index']]
            survivors = [item for item in active if item != index]
            trials = [fit(survivors, seed) for seed in seeds]
            verdict = deletion_verdict(current, trials, original, survivors, atol=atol, rtol=rtol)
            round_log['trials'].append({
                'removed_candidate': index, 'fit_keys': [row['key'] for row in trials],
                **verdict,
            })
            if checkpoint:
                checkpoint({'completed_rounds': rounds, 'current_round': round_log,
                            'confirmed_removed': removed, 'trial_in_progress': False})
            if verdict['accept_removal']:
                removed.append(index)
                active, current = survivors, trials
                accepted = True
                break
        rounds.append(round_log)
        if not accepted:
            break
    result = {
        'kept_indices': active, 'removed_indices': removed, 'rounds': rounds,
        'primary_fit': current[0], 'reference_fits': current,
        'original_fits': original, 'primary_seed': seeds[0],
        'fallback': None if removed else 'exact_vanilla_checkpoint',
        'selector': 'unanimous_checkpoint_TracIn_then_matched_deletion_refits',
        'default_action': 'keep', 'benchmark_answers_used': False,
        'heldout_labels_used': False, 'factual_nonregression_guaranteed': False,
        'original_reference_protected_per_trajectory': True,
    }
    if checkpoint:
        checkpoint(result)
    return result
