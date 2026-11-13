"""Support figure: why/how DualJSD beats vanilla SD (Qwen3-4B, yesterday's paired campaigns) + cost."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INK, INK2, GRID, SURFACE = '#22211f', '#5c5b57', '#e4e3df', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SEQ = LinearSegmentedColormap.from_list('seq', ['#f4f8fe', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
DS = ['longhealth', 'mtob_ke', 'qasper']
DSN = {'longhealth': 'LongHealth', 'mtob_ke': 'MTOB', 'qasper': 'QASPER'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                     'font.size': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 7.5, 'axes.titleweight': 'normal'})

tok = pd.read_csv(HERE / 'tokens.csv')
cells = pd.read_csv(HERE / 'cells.csv')
timing = json.loads((HERE / 'timing.json').read_text())
stats = {}

fig = plt.figure(figsize=(6.9, 5.0))
gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.62)

# (a) where targets change: E confidence x dataset (specific data)
ax = fig.add_subplot(gs[0, 0])
spec = tok[tok.arm == 'specific'].copy()
bins = [0, .5, .8, .95, 1.0001]; labels = ['<0.5', '0.5–0.8', '0.8–0.95', '>0.95']
spec['b'] = pd.cut(spec.pe, bins, labels=labels)
rt = spec.groupby(['b', 'dataset'], observed=True).retarget.mean().unstack()[DS].reindex(labels[::-1])
wt = spec.groupby(['b', 'dataset'], observed=True).w.mean().unstack()[DS].reindex(labels[::-1])
share = spec.groupby(['b', 'dataset'], observed=True).size().unstack()[DS].reindex(labels[::-1])
share = share / share.sum()
stats['a'] = dict(retarget=rt.round(4).to_dict(), weight=wt.round(3).to_dict(), share=share.round(3).to_dict())
im = ax.imshow(rt.values * 100, cmap=SEQ, vmin=0, vmax=50, aspect='auto')
for i in range(rt.shape[0]):
    for j in range(rt.shape[1]):
        v = rt.values[i, j] * 100
        ax.text(j, i - .14, f'{v:.0f}%' if v >= 1 else f'{v:.1f}%', ha='center', va='center', fontsize=7.5,
                color=SURFACE if v > 28 else INK)
        ax.text(j, i + .22, f'w={wt.values[i, j]:.2f}', ha='center', va='center', fontsize=6,
                color=SURFACE if v > 28 else INK2)
ax.set_xticks(range(3)); ax.set_xticklabels([DSN[d] for d in DS], fontsize=7)
ax.set_yticks(range(4)); ax.set_yticklabels(labels[::-1])
ax.set_ylabel('Evidence-view confidence $P_E$(top-1)')
ax.set_xlabel('specific data; cell = % re-targeted, mean weight $w$', fontsize=6, color=INK2)
ax.set_title('(a) DualJSD rewrites the target only where $E$ is unsure', loc='left')
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=5.5)
cb.set_label('top-1 re-targeted by $W$ (%)', fontsize=5.8); cb.outline.set_visible(False)
for s in ax.spines.values(): s.set_visible(False)

# (b) what tokens are down-weighted
ax = fig.add_subplot(gs[0, 1])
classes = ['number', 'capitalized/entity', 'other word', 'function word', 'punct/format']
cname = {'number': 'numbers', 'capitalized/entity': 'names / entities', 'other word': 'content words',
         'function word': 'function words', 'punct/format': 'punct. / format'}
low = spec.assign(low=spec.w < .7).groupby(['cls', 'dataset']).low.mean().unstack()[DS].reindex(classes)
stats['b'] = low.round(3).to_dict()
im = ax.imshow(low.values * 100, cmap=SEQ, vmin=0, vmax=35, aspect='auto')
for i in range(low.shape[0]):
    for j in range(low.shape[1]):
        v = low.values[i, j] * 100
        ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=7.5, color=SURFACE if v > 20 else INK)
ax.set_xticks(range(3)); ax.set_xticklabels([DSN[d] for d in DS], fontsize=7)
ax.set_yticks(range(len(classes))); ax.set_yticklabels([cname[c] for c in classes])
ax.set_title('(b) Facts kept on LongHealth/QASPER, diluted on MTOB', loc='left')
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=5.5)
cb.set_label('tokens with weight $w<0.7$ (%)', fontsize=5.8); cb.outline.set_visible(False)
for s in ax.spines.values(): s.set_visible(False)

# (c) outcome: (wins - losses) / cells vs SD on the same data
ax = fig.add_subplot(gs[1, 0])
cells['wl'] = np.sign(cells.gain.round(9))
order = [('specific', d) for d in DS] + [('agnostic', d) for d in DS]
budgets = ['r05', 'r10', 'r15', 'r20']
mat = np.array([[cells[(cells.arm == a) & (cells.dataset == d) & (cells.budget == b)].wl.mean() for b in budgets]
                for a, d in order])
stats['c'] = {f'{a}/{d}': dict(zip(budgets, np.round(mat[i], 3).tolist())) for i, (a, d) in enumerate(order)}
im = ax.imshow(mat, cmap=DIV, vmin=-.75, vmax=.75, aspect='auto')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f'{mat[i, j]:+.2f}', ha='center', va='center', fontsize=7.2,
                color=SURFACE if abs(mat[i, j]) > .45 else INK)
ax.axhline(2.5, color=SURFACE, lw=2)
ax.set_xticks(range(4)); ax.set_xticklabels(['5%', '10%', '15%', '20%'], fontsize=6)
ax.set_yticks(range(6)); ax.set_yticklabels([f'{DSN[d]}, {"specific" if a == "specific" else "agnostic"}' for a, d in order])
ax.set_xlabel('Native KV budget')
ax.set_title('(c) Paired outcome vs SD on the same data (win − loss rate)', loc='left')
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=5.5)
cb.set_label('SD better  ←   →  DualJSD better', fontsize=6); cb.outline.set_visible(False)
for s in ax.spines.values(): s.set_visible(False)

# (d) cost per cell (same GPU type: per-update time is identical by construction and measured)
ax = fig.add_subplot(gs[1, 1])
t = pd.DataFrame(timing['cells'])
t['ps'] = t.dj_sec / t.dj_updates
score = pd.DataFrame(timing['scoring']).T
score['ds'] = [k.split('/')[0] for k in score.index]
rows = []
for d in DS:
    g = t[(t.dataset == d) & (t.arm == 'specific')]
    ps = g.ps.median()
    sd_train = (g.sd_updates * ps).mean()
    dj_train = (g.dj_updates * ps).mean()
    sc = score[score.ds == d].seconds.astype(float).median() / 4  # one scoring per request, shared by 4 budgets
    rows.append(dict(dataset=d, sd_train=sd_train, dj_train=dj_train, scoring_per_cell=sc,
                     sd_updates=g.sd_updates.mean(), dj_updates=g.dj_updates.mean(), sec_per_update=ps,
                     scoring_per_request=score[score.ds == d].seconds.astype(float).median()))
cost = pd.DataFrame(rows)
stats['d'] = cost.round(2).to_dict('records')
x = np.arange(3); wbar = .36
ax.bar(x - wbar / 2 - .01, cost.sd_train, wbar, color=ORANGE, label='SD: training', zorder=3)
ax.bar(x + wbar / 2 + .01, cost.dj_train, wbar, color=BLUE, label='DualJSD: training', zorder=3)
ax.bar(x + wbar / 2 + .01, cost.scoring_per_cell, wbar, bottom=cost.dj_train, color='#86b6ef',
       label='DualJSD: two-view scoring', zorder=3, edgecolor=SURFACE, lw=.6)
for i, r in cost.iterrows():
    ax.text(i - wbar / 2, r.sd_train * 1.04, f'{r.sd_train:.0f}s', ha='center', va='bottom', fontsize=6.5, color=INK2)
    tot = r.dj_train + r.scoring_per_cell
    ax.text(i + wbar / 2, tot * 1.04, f'{tot:.0f}s', ha='center', va='bottom', fontsize=6.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([DSN[d] for d in DS], fontsize=7)
ax.set_ylabel('Seconds per cell (specific data)')
ax.set_ylim(0, (cost.dj_train + cost.scoring_per_cell).max() * 1.8)
ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc='upper left', handlelength=1.0)
ax.set_title('(d) Time per cell: fewer updates + one scoring pass', loc='left')

for ext in ('pdf', 'png', 'svg'):
    fig.savefig(HERE / f'fig_dualjsd_insights.{ext}', bbox_inches='tight', dpi=300)
(HERE / 'insights_stats.json').write_text(json.dumps(stats, indent=1, default=float))
print(cost.round(2).to_string())
