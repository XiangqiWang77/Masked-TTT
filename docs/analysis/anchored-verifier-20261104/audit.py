#!/usr/bin/env python3
"""Replay a label-free verifier on previously generated quote checks."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
CROSS=Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102')
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO))
from experiments.anchored_short_verifier.verifier import accepted
from masked_ttt.self_study import Exercise


def main():
    snapshot=json.loads((CROSS/'query_cartridges_format_audit_20261104/audit.json').read_text())
    folders={'gpt':CROSS/'query_cartridges_gpt_olmo1000_20261103/cases',
             'qwen':CROSS/'query_cartridges_qwen_olmo1000_20261104/cases'}
    models={}
    for model in ('qwen','gpt'):
        case_keys=[row['case'] for row in snapshot['models'][model]['rows']]
        if len(case_keys)!=len(set(case_keys)):raise ValueError('Duplicate frozen case identifier')
        totals=Counter();cases_with_additional=0
        for key in case_keys:
            result=json.loads((folders[model]/key/'weight_0.00.json').read_text())
            selected=set(result['selection']['selected_chunks']);added=False
            for item in result['self_study']['checks']:
                totals['checked']+=1
                if item['source_chunk'] not in selected:raise ValueError('Practice quote was not selected')
                exercise=Exercise(item['question'],item['answer'],item['evidence'],item['source_chunk'])
                keep,reason=accepted(exercise,item['verifier_answer'])
                if item['accepted']:
                    if not keep or reason!='original_exact':raise ValueError('Original accepted check changed')
                    totals['previously_verified']+=1
                elif keep and reason=='verified_anchored_subspan':
                    totals['additional_verifier_accepts']+=1;added=True
            if added:cases_with_additional+=1
        models[model]=dict(cases=len(case_keys),**totals,cases_with_additional=cases_with_additional)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source='fixed 2026-11-04 01:32 UTC 221-Qwen / 264-GPT paired snapshot',
        only_saved_generated_questions_quotes_and_independent_verifier_answers_used=True,
        benchmark_answers_and_correctness_used=False,online_api_calls=0,
        note='Replayed verifier decisions only; added practice may still fail cache-boundary checks; no accuracy claim.',
        models=models)
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(models,indent=2))


if __name__=='__main__':main()
