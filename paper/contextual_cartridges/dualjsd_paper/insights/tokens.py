"""Token-level view of what DualJSD changes vs the SD target (yesterday's Qwen campaign).

Per answer token t of each saved trajectory:
  E top-1 prob (SD target, from the saved Teacher top-k),
  w_t = sum of stored w*M entries (>=0.99 of M's mass kept, so w is recovered to <1%),
  M top-1 (after mixing with the whole-document view) and whether it differs from E's top-1,
  token class (number / capitalized entity / function word / punctuation-format / other word).
"""
import csv, glob, json, math, pickle, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/da839/Query_Cart')
from transformers import AutoTokenizer
from cartridges.structs import read_conversations
DUAL = Path('/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_dual_view_jsd_20261112_v1')
tok = AutoTokenizer.from_pretrained('/home/da839/Stack_TTT/models/Qwen3-4B-Instruct-2507')
plan = {(r['dataset'], r['case']): r for r in json.loads((DUAL / 'plan.json').read_text())}
STOP = set('the a an of to in on at for and or is are was were be been by with as that this it its from which who whom what when where how not no than then there their they he she his her we you i'.split())
def cls(s):
    t = s.strip()
    if not t: return 'whitespace/format'
    if re.search(r'\d', t): return 'number'
    if not re.search(r'[A-Za-z]', t): return 'punct/format'
    if t.lower() in STOP: return 'function word'
    if t[0].isupper(): return 'capitalized/entity'
    return 'other word'
rows = []
for audit in sorted(DUAL.glob('dual_targets/*/*/audit.json')):
    d, c = audit.parts[-3], audit.parts[-2]
    conv = pickle.loads((audit.parent / 'targets.pkl').read_bytes())
    files = json.loads(Path(plan[(d, c)]['synthesis_manifest']).read_text())['data_files'] if plan[(d, c)].get('synthesis_manifest') else plan[(d, c)]['data_files']
    orig = {'specific': read_conversations(files['specific_targets']), 'agnostic': read_conversations(files['agnostic_targets'])[:15]}
    for arm, convs in conv.items():
        for j, (new, old) in enumerate(zip(convs, orig[arm])):
            a_new, a_old = new.messages[-1].top_logprobs, old.messages[-1].top_logprobs
            ids = old.messages[-1].token_ids
            T = len(ids)
            for t in range(T):
                mo = a_old.token_idx == t; mn = a_new.token_idx == t
                lp_o, id_o = a_old.logprobs[mo], a_old.token_id[mo]
                lp_n, id_n = a_new.logprobs[mn], a_new.token_id[mn]
                if len(lp_o) == 0 or len(lp_n) == 0: continue
                pe = float(np.exp(lp_o).max()); e_top = int(id_o[np.argmax(lp_o)])
                p_n = np.exp(lp_n); w = float(min(1.0, p_n.sum()))
                m_top = int(id_n[np.argmax(lp_n)]); m_p = float(p_n.max() / max(w, 1e-9))
                piece = tok.decode([ids[t]])
                rows.append(dict(dataset=d, case=c, arm=arm, traj=j, t=t, token=piece, cls=cls(piece),
                                 pe=pe, w=w, m_top_p=m_p, retarget=int(m_top != e_top),
                                 answer_is_e_top=int(ids[t] == e_top)))
out = Path(__file__).with_name('tokens.csv')
with out.open('w', newline='') as h:
    wr = csv.DictWriter(h, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(len(rows), 'tokens ->', out)
