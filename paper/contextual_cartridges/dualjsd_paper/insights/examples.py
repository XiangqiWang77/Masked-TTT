"""Pick real training tokens and recover P_E, P_W, and the DualJSD target for each.

P_E: saved evidence-view top-20 (the SD target). Stored DualJSD entries are log(w*M) on the
union support (>=0.99 of M's mass), so M = entries / w and P_W = 2M - P_E on that support
(P_E outside its saved top-20 is below the 20th probability and is taken as 0).
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/da839/Query_Cart')
from transformers import AutoTokenizer
from cartridges.structs import read_conversations

HERE = Path(__file__).resolve().parent
DUAL = Path('/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_dual_view_jsd_20261112_v1')
tok = AutoTokenizer.from_pretrained('/home/da839/Stack_TTT/models/Qwen3-4B-Instruct-2507')
plan = {(r['dataset'], r['case']): r for r in json.loads((DUAL / 'plan.json').read_text())}
t = pd.read_csv(HERE / 't_small.csv', keep_default_na=False) if (HERE / 't_small.csv').exists() else None
if t is None:
    t = pd.read_csv(HERE / 'tokens.csv', keep_default_na=False)
    t = t[(t.arm == 'specific') & t.dataset.isin(['qasper', 'longhealth'])]
    t.to_csv(HERE / 't_small.csv', index=False)
alpha = t.token.str.strip().str.match(r'^[A-Za-z0-9]{3,}$')
picks = {
    'resolve': t[alpha & (t.pe < .45) & (t.retarget == 1) & (t.m_top_p > .55) & (t.w > .45)],
    'conflict': t[alpha & (t.pe > .85) & (t.w < .35)],
    'agree': t[alpha & (t.pe > .97) & (t.w > .985)],
}
out = {}
cache = {}
for name, cand in picks.items():
    cand = cand.sample(frac=1, random_state=3)
    for _, row in cand.iterrows():
        key = (row.dataset, row.case)
        if key not in cache:
            conv = pickle.loads((DUAL / 'dual_targets' / row.dataset / row.case / 'targets.pkl').read_bytes())['specific']
            orig = read_conversations(plan[key]['data_files']['specific_targets'])
            cache[key] = (conv, orig)
        conv, orig = cache[key]
        new, old = conv[row.traj].messages[-1], orig[row.traj].messages[-1]
        tt = int(row.t)
        e_ids = old.top_logprobs.token_id[old.top_logprobs.token_idx == tt]
        e_p = np.exp(old.top_logprobs.logprobs[old.top_logprobs.token_idx == tt])
        n_ids = new.top_logprobs.token_id[new.top_logprobs.token_idx == tt]
        n_p = np.exp(new.top_logprobs.logprobs[new.top_logprobs.token_idx == tt])
        w = float(min(1., n_p.sum()))
        m = {int(i): float(p) / w for i, p in zip(n_ids, n_p)}
        pe = {int(i): float(p) for i, p in zip(e_ids, e_p)}
        ids = sorted(set(m) | set(pe), key=lambda i: -(m.get(i, 0) + pe.get(i, 0)))[:6]
        pw = {i: max(0., 2 * m.get(i, 0.) - pe.get(i, 0.)) for i in ids}
        if name == 'conflict' and max(pw.values()) < .5:
            continue
        if name == 'resolve' and max(pw.values()) < .6:
            continue
        context = tok.decode(old.token_ids[max(0, tt - 14):tt]).replace('\n', ' ')
        out[name] = dict(dataset=row.dataset, case=row.case, traj=int(row.traj), t=tt, w=w,
                         question=orig[row.traj].messages[0].content[:160], context=context,
                         candidates=[dict(token=tok.decode([i]), p_e=pe.get(i, 0.), p_w=pw[i], target=w * m.get(i, 0.))
                                     for i in ids],
                         written=tok.decode([old.token_ids[tt]]))
        break
(HERE / 'examples.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
for k, v in out.items():
    print(k, v['dataset'], 'w=%.2f' % v['w'], '| ...' + v['context'][-70:], '->', [(c['token'], round(c['p_e'], 2), round(c['p_w'], 2), round(c['target'], 2)) for c in v['candidates'][:4]])
