"""Matched comparison: Whole KV, Evidence Teacher, SnapKV, vanilla/specific Cartridges at 5/10/15/20%.

LongHealth is rescored from saved predictions with one scorer (actual-output
unique-option extraction). MTOB = chrF (per case). QASPER = reference log-PPL (lower better).
"""
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/da839/Query_Cart'); sys.path.insert(0, str(ROOT))
from experiments.run_figure4_contextual import load_cases, longhealth_score  # noqa: E402

DATA = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/contextual_cartridges_figure4_datasets_20261110'
NINE = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/contextual_nine_methods_devel6_20261111_v1'
SNAP = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cartridges_ratio_baselines_20261112_v1/snapkv/results/snapkv'
FAST = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_fast_20261112_v2/results'
FASTKV = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cartridges_fastkv_ratio_continuation_20261113_v1/results/fast_kv'
DUAL = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_dual_view_jsd_20261112_v1/results'
OUT = Path(__file__).resolve().parent
DATASETS = ('longhealth', 'mtob_ke', 'qasper')
BUDGETS = ('r05', 'r10', 'r15', 'r20')
cases = {d: {c['id']: c for c in load_cases(d, DATA)} for d in DATASETS}


def score(dataset, path):
    r = json.loads(Path(path).read_text())
    if dataset == 'longhealth':
        return float(longhealth_score(r.get('prediction') or '', cases[dataset][r['case'] if r.get('case') in cases[dataset] else Path(path).stem]))
    return float(r['metric_value'])


def collect(pattern):
    out = {}
    for path in glob.glob(pattern):
        out[Path(path).stem] = path
    return out


methods_fixed = {'Whole KV': f'{NINE}/icl/results/icl/{{d}}/full/*.json',
                 'Evidence Teacher': f'{NINE}/evidence_teacher/results/evidence_teacher/{{d}}/evidence/*.json'}
methods_budget = {'SnapKV': f'{SNAP}/{{d}}/{{b}}/*.json',
                  'Vanilla Cartridges (agnostic SD)': f'{FAST}/vanilla_cartridges/{{d}}/{{b}}/*.json',
                  'Specific Cartridges (specific SD)': f'{FAST}/specific_sd/{{d}}/{{b}}/*.json',
                  'Contextual (specific + ED-JSD)': f'{FAST}/contextual_cartridges/{{d}}/{{b}}/*.json',
                  'Agnostic + DualJSD (running)': f'{DUAL}/agnostic_dual_jsd/{{d}}/{{b}}/*.json',
                  'Specific + DualJSD (running)': f'{DUAL}/specific_dual_jsd/{{d}}/{{b}}/*.json',
                  'FastKV (attention matching)': f'{FASTKV}/{{d}}/{{b}}/*.json'}
CORE = ['Whole KV', 'Evidence Teacher', 'SnapKV', 'Vanilla Cartridges (agnostic SD)',
        'Specific Cartridges (specific SD)']

rows, summary = [], {}
for d in DATASETS:
    fixed = {m: collect(p.format(d=d)) for m, p in methods_fixed.items()}
    for b in BUDGETS:
        files = dict(fixed)
        files.update({m: collect(p.format(d=d, b=b)) for m, p in methods_budget.items()})
        core_ids = set.intersection(*(set(files[m]) for m in CORE))
        entry = {'matched_core_cases': len(core_ids), 'methods': {}}
        for m, f in files.items():
            ids = sorted(core_ids & set(f))
            if not ids:
                entry['methods'][m] = {'n': 0, 'available_total': len(f)}
                continue
            vals = [score(d, f[i]) for i in ids]
            entry['methods'][m] = {'n': len(ids), 'mean': st.mean(vals), 'available_total': len(f)}
            for i, v in zip(ids, vals):
                rows.append({'dataset': d, 'budget': b, 'case': i, 'method': m, 'value': v})
        summary[f'{d}/{b}'] = entry
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
with (OUT / 'per_case.csv').open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=['dataset', 'budget', 'case', 'method', 'value']); w.writeheader(); w.writerows(rows)
for key, entry in summary.items():
    print(key, 'n_core', entry['matched_core_cases'])
    for m, v in entry['methods'].items():
        print(f"   {m:36s} n={v['n']:3d} mean={v.get('mean', float('nan')):.3f} (available {v['available_total']})")
