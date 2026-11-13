"""Two angles on DualJSD as distributions.

Fig 1 (teacher side): real training tokens - evidence view P_E, whole-document view P_W, and the
DualJSD target w*M that the Cartridge is trained on (SD trains on P_E alone).
Fig 2 (student side): distribution of test scores of the trained Cartridges, SD vs DualJSD.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
INK, INK2, GRID, SURFACE = '#22211f', '#5c5b57', '#e4e3df', '#ffffff'
GREEN, VIOLET, BLUE, ORANGE = '#1baf7a', '#6250d6', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                     'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 8, 'axes.linewidth': .7})


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)


# ---------------- Figure 1: the two views and the DualJSD target ----------------
ex = json.loads((HERE / 'examples.json').read_text())
cases = [('resolve', 'E is unsure, W is confident\n→ target follows the agreeing mass'),
         ('conflict', 'E and W confidently disagree\n→ $w\\approx0$: token is not learned'),
         ('agree', 'E and W agree\n→ same target as SD ($w\\approx1$)')]
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.75), gridspec_kw={'wspace': .38})
for ax, (key, title) in zip(axes, cases):
    clean(ax)
    e = ex[key]
    cand = e['candidates'][:3] if key != 'agree' else e['candidates'][:1] + [dict(token='(other)', p_e=0, p_w=0, target=0)]
    labels = [c['token'].strip() or repr(c['token']) for c in cand]
    x = np.arange(len(cand)); bw = .26
    for off, k, color, lab in ((-bw, 'p_e', GREEN, 'evidence view $P_E$ (= SD target)'),
                               (0, 'p_w', VIOLET, 'whole-document view $P_W$'),
                               (bw, 'target', BLUE, 'DualJSD target $w\\cdot M$')):
        vals = [c[k] for c in cand]
        ax.bar(x + off, vals, bw * .92, color=color, label=lab, zorder=3)
        for xi, v in zip(x + off, vals):
            if v >= .05:
                ax.text(xi, v + .02, f'{v:.2f}', ha='center', va='bottom', fontsize=5.3, color=INK2, rotation=90 if key == 'resolve' else 0)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.2, family='DejaVu Sans')
    ax.set_ylim(0, 1.18); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_title(title, loc='left', fontsize=7.6)
    ctx = e['context'].strip()[-30:]
    ax.text(0, -.24, f'context: “…{ctx} ▮”', transform=ax.transAxes, fontsize=5.6, color=INK2, family='DejaVu Sans')
    ax.text(.98, .97, f'$w$ = {e["w"]:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=INK)
axes[0].set_ylabel('next-token probability')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(.5, 1.07))
for ext in ('pdf', 'png', 'svg'):
    fig.savefig(HERE / f'fig_dualjsd_two_views.{ext}', bbox_inches='tight', dpi=300)
plt.close(fig)

# ---------------- Figure 2: test-score distributions of trained Cartridges ----------------
cells = pd.read_csv(HERE / 'cells.csv')
spec = cells[cells.arm == 'specific']
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.45), gridspec_kw={'wspace': .42})
stats = {}
for ax, d, xl, better in ((axes[0], 'qasper', 'test reference log-PPL', '↓'),
                          (axes[1], 'mtob_ke', 'test chrF', '↑')):
    clean(ax)
    g = spec[spec.dataset == d]
    lo, hi = np.percentile(np.r_[g.sd, g.dual], [0, 100]); pad = (hi - lo) * .08
    grid = np.linspace(lo - pad, hi + pad, 300)
    for key, color, lab in (('sd', ORANGE, 'SD'), ('dual', BLUE, 'DualJSD')):
        v = g[key].values
        dens = gaussian_kde(v, bw_method=.35)(grid)
        ax.fill_between(grid, 0, dens, color=color, alpha=.25, lw=0)
        ax.plot(grid, dens, color=color, lw=1.5, label=lab)
        ax.axvline(v.mean(), color=color, lw=1, ls=(0, (3, 2)))
        stats[f'{d}/{key}'] = dict(mean=float(v.mean()), median=float(np.median(v)), n=len(v))
    ax.set_xlabel(f'{xl} {better}'); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.yaxis.grid(False)
    ax.set_title('QASPER: shifts to better' if d == 'qasper' else 'MTOB: shifts to worse', loc='left')
    ax.text(.98, .97, f"mean SD {g.sd.mean():.2f}\nmean DualJSD {g.dual.mean():.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=6.3, color=INK2)
ax = axes[2]; clean(ax)
g = spec[spec.dataset == 'longhealth']
acc = g.groupby('case')[['sd', 'dual']].mean()          # per request, accuracy over 4 budgets
levels = np.array([0, .25, .5, .75, 1.0])
for off, key, color, lab in ((-.045, 'sd', ORANGE, 'SD'), (.045, 'dual', BLUE, 'DualJSD')):
    counts = [(np.isclose(acc[key], lv)).sum() for lv in levels]
    ax.bar(levels + off, counts, .085, color=color, label=lab, zorder=3)
    stats[f'longhealth/{key}'] = dict(mean=float(acc[key].mean()), hist=dict(zip(map(str, levels), map(int, counts))))
ax.set_xticks(levels); ax.set_xticklabels(['0/4', '1/4', '2/4', '3/4', '4/4'])
ax.set_xlabel('budgets correct per question')
ax.text(.02, .97, f"mean SD {acc.sd.mean()*100:.0f}%\nmean DualJSD {acc.dual.mean()*100:.0f}%", transform=ax.transAxes, va='top', fontsize=6.3, color=INK2)
ax.set_ylabel('questions'); ax.set_title('LongHealth: more at 4/4', loc='left')
h, l = axes[2].get_legend_handles_labels()
fig.legend(h, l, loc='upper right', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(.98, 1.07))
fig.text(.07, 1.0, 'Trained Cartridges on the real test questions (specific data, 15 requests × 4 budgets)',
         fontsize=8, color=INK)
for ext in ('pdf', 'png', 'svg'):
    fig.savefig(HERE / f'fig_dualjsd_outcome_dist.{ext}', bbox_inches='tight', dpi=300)
(HERE / 'insights4_stats.json').write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=0))
