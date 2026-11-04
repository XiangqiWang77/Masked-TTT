#!/usr/bin/env python3
"""Summarize accepted OLMo update signals without model or API calls."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
CASES=Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102/query_cartridges_20261103/runs/olmo3-7b-babilong/cases')
DEST=ROOT/'docs/analysis/qc-unified-motivation-20261104/data/gradient_signals.json'


def main():
    paths=list(CASES.glob('*/masked_ttt.json'))
    if len(paths)!=1000:
        raise ValueError('Expected 1,000 OLMo case logs')
    records=[]
    for path in paths:
        data=json.loads(path.read_text())
        if data['key']!=path.parent.name or not data['backbone_unchanged']:
            raise ValueError('Unexpected case or mutable backbone')
        steps=data['training']['steps']
        if not steps or any(not math.isfinite(s['gain_gradient_norm']) or
                            not math.isfinite(s['value_gradient_norm']) for s in steps):
            raise ValueError('Missing or invalid adaptation signals')
        records.append(dict(accepted=data['training']['accepted_steps'],
            has_gain_gradient=any(s['gain_gradient_norm']>0 for s in steps),
            has_value_gradient=any(s['value_gradient_norm']>0 for s in steps),
            first_gain_gradient=steps[0]['gain_gradient_norm'],
            first_value_gradient=steps[0]['value_gradient_norm']))
    result=dict(cases=len(records),cases_with_accepted_step=sum(x['accepted']>0 for x in records),
        cases_with_nonzero_gain_gradient=sum(x['has_gain_gradient'] for x in records),
        cases_with_nonzero_value_gradient=sum(x['has_value_gradient'] for x in records),
        median_first_gain_gradient=statistics.median(x['first_gain_gradient'] for x in records),
        median_first_value_gradient=statistics.median(x['first_value_gradient'] for x in records),
        interpretation='nonzero gradients show that training signals reached both parameter groups; they do not measure causal answer gains',
        source='original OLMo 1,000-case Query Cartridge training logs',
        new_model_calls=0,online_api_calls=0)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
