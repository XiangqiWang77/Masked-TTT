"""Paired contribution estimates; this module never participates in training."""
from collections import defaultdict
import random


def paired_cluster_delta(rows, baseline, target, *, replicates=4000, seed=20261109):
    """Resample source questions with all their available lengths together.

    Sampling is stratified by task. Confidence intervals are descriptive until
    the frozen evaluation partition has finished; partial rows are excluded.
    """
    pairs = [r for r in rows if baseline in r['scores'] and target in r['scores']]
    if not pairs:
        return {'n': 0, 'source_questions': 0, 'delta_pp': None, 'ci95_pp': None}
    clusters = defaultdict(lambda: [0, 0])
    for row in pairs:
        key = (row['task'], row.get('source_question', row['case']))
        clusters[key][0] += row['scores'][target] - row['scores'][baseline]
        clusters[key][1] += 1
    strata = defaultdict(list)
    for (task, _), value in clusters.items():
        strata[task].append(tuple(value))
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        numerator = denominator = 0
        for task in sorted(strata):
            values = strata[task]
            for _ in values:
                difference, count = values[rng.randrange(len(values))]
                numerator += difference
                denominator += count
        draws.append(100 * numerator / denominator)
    draws.sort()
    return {'n': len(pairs), 'source_questions': len(clusters),
            'delta_pp': 100 * sum(r['scores'][target] - r['scores'][baseline]
                                  for r in pairs) / len(pairs),
            'ci95_pp': [draws[int(.025 * (replicates - 1))],
                        draws[int(.975 * (replicates - 1))]],
            'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
            'sampling_unit': 'source_question_all_document_lengths_together',
            'stratification': 'BABILong_task',
            'few_source_questions_warning': len(clusters) < 20}


def summarize_partitions(rows, arm_names, partition_metadata):
    """Expose prior evaluation, rather than relabeling it as pristine holdout."""
    complete = [r for r in rows if all(a in r['scores'] for a in arm_names)]
    definitions = {
        'query_guided_data': ('vanilla_syns_vanilla_sd', 'specific_syns_vanilla_sd'),
        'selective_SD_with_vanilla_data': ('vanilla_syns_vanilla_sd', 'vanilla_syns_selective_sd'),
        'selective_SD_with_guided_data': ('specific_syns_vanilla_sd', 'specific_syns_selective_sd'),
    }
    result = {}
    for name in ('development', 'previously_evaluated_source_holdout', 'unseen_source_holdout'):
        planned = [r for r in rows if r.get('evaluation_partition') == name]
        observed = [r for r in complete if r.get('evaluation_partition') == name]
        effects = {label: paired_cluster_delta(observed, base, target)
                   for label, (base, target) in definitions.items()
                   if base in arm_names and target in arm_names}
        finished = bool(planned) and len(observed) == len(planned)
        positive = bool(effects) and all(e['delta_pp'] is not None and e['delta_pp'] > 0
                                         for e in effects.values())
        intervals = bool(effects) and all(e['ci95_pp'] and e['ci95_pp'][0] > 0
                                          for e in effects.values())
        result[name] = {'planned_cases': len(planned), 'completed_cases': len(observed),
                        'complete': finished, 'contributions': effects,
                        'all_three_point_estimates_positive': positive,
                        'all_three_individual_ci95_lower_bounds_positive': intervals,
                        'independent_confirmation': name == 'unseen_source_holdout',
                        'claims_ready': finished and positive and intervals}
    return {'partitions': result,
            'prior_evaluation_audit': partition_metadata.get('provenance', {}),
            'intervals_are_individual_not_simultaneous': True,
            'partial_results_are_exploratory': True,
            'no_benchmark_references_used_in_training': True}
