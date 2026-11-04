#!/usr/bin/env python3
"""Prepare a fixed, paired OLMo diagnostic sample from existing annotations."""
import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import random

WORKSPACE = Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102/query_cartridges_20261103')
SNAPSHOT = Path(__file__).resolve().parents[2] / 'docs/analysis/olmo-motivations-20261104/data_snapshot.csv'
LENGTHS = (8192, 16384, 32768, 65536)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--questions', type=int, default=12)
    args = parser.parse_args()
    if args.questions < 1:
        raise ValueError('The number of source questions must be positive')
    spec = importlib.util.spec_from_file_location('saved_support_audit', WORKSPACE/'audit_babilong_selected_support.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    units, matches = audit.source_matches(json.loads(audit.RESERVATION.read_text()))
    panel = {r['case']: r for r in json.loads((audit.ORIGINAL/'fresh_confirmation.json').read_text())}
    config = json.loads((WORKSPACE/'runs/olmo3-7b-babilong/protocol.json').read_text())
    index = {r['key']: r for r in json.loads(Path(config['index']).read_text())}
    with SNAPSHOT.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000:
        raise ValueError('Expected complete OLMo snapshot')
    groups = defaultdict(dict)
    for row in rows:
        if row['working_document'] == 'complete':
            groups[row['source_question']][int(row['context_tokens'])] = row
    eligible = {k: v for k, v in groups.items() if set(v) == set(LENGTHS)}
    if len(eligible) < args.questions:
        raise ValueError('Not enough questions have all four original contexts')
    by_task = defaultdict(list)
    for question, cases in eligible.items():
        by_task[cases[LENGTHS[0]]['task']].append(question)
    rng = random.Random(args.seed)
    chosen = []
    if args.questions >= len(by_task):
        for task in sorted(by_task):
            chosen.append(rng.choice(sorted(by_task[task])))
    remaining = sorted(set(eligible) - set(chosen))
    chosen.extend(rng.sample(remaining, args.questions-len(chosen)))
    chosen.sort()
    cases = []
    with Path(config['source_cases']).open('rb') as source:
        for question in chosen:
            for length in LENGTHS:
                row = eligible[question][length]
                entry = index[row['case']]
                source.seek(entry['offset'])
                case = json.loads(source.read(entry['bytes']))
                if case['key'] != row['case']:
                    raise ValueError('Source case identity mismatch')
                unit = panel[row['case']]['sampling_unit']
                if case['query'].strip() != units[unit]['question'].strip():
                    raise ValueError('Question and annotation differ')
                candidate_spans = []
                for annotated in matches[unit]:
                    facts, wanted = annotated['facts'], set(annotated['support'])
                    occurrences = audit.ordered_occurrences(case['document'], [text for _, text in facts])
                    spans = []
                    for (number, _), positions in zip(facts, occurrences):
                        if number in wanted:
                            if len(positions) != 1:
                                raise ValueError('Ambiguous support fact position')
                            spans.append(positions[0])
                    candidate_spans.append(tuple(sorted(spans)))
                if not candidate_spans or len(set(candidate_spans)) != 1:
                    raise ValueError('Ambiguous support annotation')
                lo, hi = float(row['fact_cover_lower']), float(row['fact_cover_upper'])
                if lo != hi:
                    raise ValueError('Selected support coverage is ambiguous')
                cases.append(dict(key=row['case'], source_question=question, task=row['task'],
                    context_tokens=length, offset=entry['offset'], bytes=entry['bytes'],
                    support_character_spans=candidate_spans[0], selected_fact_coverage=lo))
    if Counter(r['context_tokens'] for r in cases) != Counter({length: args.questions for length in LENGTHS}):
        raise ValueError('Paired pilot is not balanced across context lengths')
    result = dict(seed=args.seed, source_question_count=len(chosen), case_count=len(cases),
        sampling='stratified by task among complete-document source questions with all four contexts; random within strata',
        only_complete_original_documents=True, source_cases=config['source_cases'],
        no_reference_answers_in_plan=True, annotation_role='post-training measurement only', cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2)+'\n')
    tmp.replace(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
