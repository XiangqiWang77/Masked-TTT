"""Join DualJSD vs SD paired cells (yesterday's Qwen campaigns) with per-request view-disagreement audits."""
import csv, glob, json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, '/home/da839/Query_Cart')
from experiments.run_figure4_contextual import load_cases, longhealth_score
DATA = '/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/contextual_cartridges_figure4_datasets_20261110'
FAST = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_fast_20261112_v2/results'
DUAL = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_dual_view_jsd_20261112_v1'
SIGN = {'longhealth': 1, 'mtob_ke': 1, 'qasper': -1}
cases = {d: {c['id']: c for c in load_cases(d, DATA)} for d in SIGN}
def score(d, path):
    r = json.loads(Path(path).read_text())
    return float(longhealth_score(r.get('prediction') or '', cases[d][r['case']])) if d == 'longhealth' else float(r['metric_value'])
rows = []
for arm, sd, dj in (('specific', 'specific_sd', 'specific_dual_jsd'), ('agnostic', 'vanilla_cartridges', 'agnostic_dual_jsd')):
    for f in glob.glob(f'{DUAL}/results/{dj}/*/*/*.json'):
        d, b, c = Path(f).parts[-3], Path(f).parts[-2], Path(f).stem
        ref = f'{FAST}/{sd}/{d}/{b}/{c}.json'
        if not Path(ref).exists():
            continue
        audit = Path(f'{DUAL}/dual_targets/{d}/{c}/audit.json')
        a = json.loads(audit.read_text())['arms'][arm]['trajectories'] if audit.exists() else []
        m = lambda k: st.mean(t[k] for t in a) if a else float('nan')
        v_dj, v_sd = score(d, f), score(d, ref)
        rows.append(dict(arm=arm, dataset=d, budget=b, case=c, dual=v_dj, sd=v_sd, gain=SIGN[d] * (v_dj - v_sd),
            mean_weight=m('mean_weight'), mean_jsd=m('mean_jsd_bits'), frac_jsd03=m('fraction_jsd_above_0.3'),
            ew_agree=m('e_w_top1_agreement'), lp_e=m('answer_token_mean_logprob_e'), lp_w=m('answer_token_mean_logprob_w'),
            ans_is_w_top1=m('answer_token_is_w_top1')))
out = Path(__file__).with_name('cells.csv')
with out.open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), 'paired cells ->', out)
