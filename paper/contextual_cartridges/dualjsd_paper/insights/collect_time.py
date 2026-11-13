import glob, json, statistics as st
from pathlib import Path
FAST = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_fast_20261112_v2/results'
DUAL = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_dual_view_jsd_20261112_v1'
rows = []
for arm, sd, dj in (('specific', 'specific_sd', 'specific_dual_jsd'), ('agnostic', 'vanilla_cartridges', 'agnostic_dual_jsd')):
    for f in glob.glob(f'{DUAL}/results/{dj}/*/*/*.json'):
        d, b, c = Path(f).parts[-3], Path(f).parts[-2], Path(f).stem
        g = f'{FAST}/{sd}/{d}/{b}/{c}.json'
        if not Path(g).exists(): continue
        a, s = json.load(open(f))['training'], json.load(open(g))['training']
        h = lambda t: [x.get('loss', x.get('training_loss')) for x in t['history'] if isinstance(x, dict)]
        rows.append(dict(arm=arm, dataset=d, budget=b, case=c,
            dj_updates=a['updates'], sd_updates=s['updates'], dj_sec=a['seconds'], sd_sec=s['seconds'],
            dj_stop=a['stop_reason'], sd_stop=s['stop_reason'],
            dj_hist=h(a), sd_hist=h(s),
            dj_red=a['learning_diagnostics']['relative_training_loss_reduction'],
            sd_red=s['learning_diagnostics']['relative_training_loss_reduction']))
scoring = {}
for au in glob.glob(f'{DUAL}/dual_targets/*/*/audit.json'):
    x = json.load(open(au)); scoring[au.split('/')[-3] + '/' + au.split('/')[-2]] = dict(seconds=x['seconds'], prefill=x['whole_view'].get('prefill_seconds'), doc_tokens=x['whole_view']['retained_document_tokens'])
json.dump(dict(cells=rows, scoring=scoring), open('timing.json', 'w'))
print(len(rows), len(scoring)); print(json.dumps(rows[0])[:600])
