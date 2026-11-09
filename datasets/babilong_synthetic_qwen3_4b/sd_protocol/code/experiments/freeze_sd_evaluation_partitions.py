"""Freeze evaluation provenance without reading answers or changing GPU inputs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.run_syn_sd_factorial import read, write


def freeze(config_path, annotations, previous_roots):
    config = read(config_path)
    destination = Path(config['output'])/'evaluation_partitions.json'
    if destination.exists():
        raise FileExistsError('Evaluation provenance is already frozen')
    families = {}
    import json
    with Path(annotations).open() as stream:
        for line in stream:
            record = json.loads(line)
            families[record['case']] = record['sampling_unit']
    previous = {p.stem for root in previous_roots
                for p in (Path(root)/'results/cases').glob('*.json')}
    observed_families = {families[key] for key in previous if key in families}
    cases = {}
    for item in config['cases']:
        key = item['key']
        seen = families[key] in observed_families
        partition = ('development' if item['analysis_split'] == 'development'
                     else 'previously_evaluated_source_holdout' if seen
                     else 'unseen_source_holdout')
        cases[key] = {'source_question': families[key], 'evaluation_partition': partition,
                      'prior_completed_case': key in previous,
                      'prior_evaluated_source_question': seen}
    counts = Counter(row['evaluation_partition'] for row in cases.values())
    value = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'provenance': {'prior_completed_case_roots': [str(p) for p in previous_roots],
                 'prior_completed_cases': len(previous),
                 'prior_evaluated_source_questions': len(observed_families),
                 'partition_counts': dict(counts),
                 'unseen_scope': 'no completed case at any length in the listed prior SD runs',
                 'all_lengths_of_a_prior_evaluated_source_marked_seen': True,
                 'prior_accuracy_values_not_read_to_construct_partition': True,
                 'training_inputs_and_original_160_840_split_unchanged': True,
                 'annotations_read_only_for_source_identity_not_answers_or_support': True},
             'cases': cases}
    write(destination, value)
    return destination, value['provenance']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--prior-root', action='append', required=True)
    args = parser.parse_args()
    destination, summary = freeze(args.config, args.annotations, args.prior_root)
    print({'path': str(destination), **summary})


if __name__ == '__main__':
    main()
