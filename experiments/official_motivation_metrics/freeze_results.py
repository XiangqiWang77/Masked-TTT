#!/usr/bin/env python3
"""Publish the complete paired diagnostic, excluding source text and answers."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'docs/analysis/official-motivation-figure-20261104'
LENGTHS = (8192, 16384, 32768, 65536)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads((args.output/'plan.json').read_text())
    status = json.loads((args.output/'figure_status.json').read_text())
    records = [json.loads(f.read_text()) for f in (args.output/'cases').glob('*.json')]
    if (not status['complete'] or status['completed'] != plan['case_count'] or
        len(records) != 48 or {r['key'] for r in records} != {r['key'] for r in plan['cases']}):
        raise ValueError('A completed paired attention experiment is required')
    if any(r['online_api_calls'] or not r['backbone_unchanged'] or
           r['attention']['reference_answers_used_for_training'] or
           not r['attention']['matched_original_question_suffix'] or
           not r['attention']['qc_first_token_matches_uninstrumented_generation'] for r in records):
        raise ValueError('A diagnostic execution contract failed')
    for row in records:
        for mode in ('whole','icl','qc'):
            item = row['attention'][mode]
            if (item['layers'] != 32 or item['answer_steps'] != row['attention']['common_reference_answer_steps'] or
                not 0 <= item['attention_mass'] <= 1):
                raise ValueError('A layer, step or attention probability is invalid')
    if Counter(r['context_tokens'] for r in records) != Counter({n:12 for n in LENGTHS}):
        raise ValueError('The attention sample is not paired across context lengths')
    sources = {name: f'question_{i:02d}' for i,name in enumerate(sorted({r['source_question'] for r in records}), 1)}
    if len(sources) != 12:
        raise ValueError('Expected exactly 12 source questions')
    rows = []
    for index,row in enumerate(sorted(records, key=lambda r:(r['source_question'],r['context_tokens'])),1):
        rows.append(dict(sample=f'case_{index:03d}', source_question=sources[row['source_question']],
            task=row['task'], context_tokens=row['context_tokens'],
            whole_target_attention_mass=row['attention']['whole']['attention_mass'],
            icl_target_attention_mass=row['attention']['icl']['attention_mass'],
            qc_target_attention_mass=row['attention']['qc']['attention_mass'],
            complete_selected_evidence=row['complete_selected_evidence'],
            answer_steps=row['attention']['common_reference_answer_steps'],
            full_support_tokens=row['attention']['full_support_token_count'],
            selected_support_tokens=row['attention']['selected_support_token_count']))
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'data').mkdir(exist_ok=True)
    (DEST/'figures').mkdir(exist_ok=True)
    with (DEST/'data/target_attention_points.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    for name in ('figure_data.csv','figure_status.json'):
        shutil.copy2(args.output/name, DEST/'data'/name)
    for ext in ('png','pdf','svg'):
        shutil.copy2(args.output/f'motivation_1_official_metrics.{ext}',
                     DEST/'figures'/f'motivation_1_official_metrics.{ext}')
    print('Published:',len(rows),'anonymized paired attention cases and three figure formats')


if __name__ == '__main__':
    main()
