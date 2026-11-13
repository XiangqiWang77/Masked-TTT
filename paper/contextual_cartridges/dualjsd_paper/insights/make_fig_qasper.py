"""QASPER only: test log-PPL distribution of Contextual Cartridges (specific + DualJSD) vs SD, Evidence ICL, Whole KV."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
INK, INK2, GRID = '#22211f', '#5c5b57', '#e4e3df'
BLUE, ORANGE, GREEN, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                     'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 8, 'axes.linewidth': .7})

cells = pd.read_csv(HERE / 'cells.csv')
ours = cells[(cells.dataset == 'qasper') & (cells.arm == 'specific')]          # 15 requests x 4 budgets
per_case = pd.read_csv('/home/da839/Query_Cart/runs/full_comparison_20261112/per_case.csv')
base = (per_case[(per_case.dataset == 'qasper') & per_case.method.isin(['Whole KV', 'Evidence Teacher'])]
        .groupby(['method', 'case']).value.first().unstack(0))
ours = ours.join(base, on='case')

panels = [('sd', 'Specific + SD', ORANGE, '(a) vs SD on the same data'),
          ('Evidence Teacher', 'Evidence ICL (no compression)', GREEN, '(b) vs the evidence teacher'),
          ('Whole KV', 'Whole KV (full document)', VIOLET, '(c) vs the whole-document teacher')]
grid = np.linspace(0, 7, 400)
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35), sharey=False, gridspec_kw={'wspace': .18})
stats = {}
for ax, (col, name, color, title) in zip(axes, panels):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    other = ours[col].values if col == 'sd' else base[col].values
    for v, c, lab in ((other, color, name), (ours.dual.values, BLUE, 'Contextual Cartridges (ours)')):
        dens = gaussian_kde(v, bw_method=.4)(grid)
        ax.fill_between(grid, 0, dens, color=c, alpha=.25, lw=0)
        ax.plot(grid, dens, color=c, lw=1.6, label=lab)
        ax.axvline(np.mean(v), color=c, lw=1, ls=(0, (3, 2)))
    better = (ours.dual < ours[col]).mean()
    stats[col] = dict(mean_other=float(np.mean(other)), mean_ours=float(ours.dual.mean()), ours_better_cells=float(better),
                      n_cells=len(ours), n_requests_other=int(len(other)))
    ax.text(.98, .97, f'mean {np.mean(other):.2f} → {ours.dual.mean():.2f}\nours lower on {better*100:.0f}% of cells',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.4, color=INK2)
    ax.set_xlim(0, 7); ax.set_xlabel('test reference log-PPL ↓')
    ax.set_title(title, loc='left')
    ax.legend(frameon=False, fontsize=6.4, loc='upper left', bbox_to_anchor=(0, -.3))
fig.text(.125, -.42, 'QASPER, Qwen3-4B; ours = query-specific data + DualJSD, 15 requests × 4 native KV budgets (5–20%); '
         'Evidence ICL / Whole KV have no budget (one value per request). Trained Cartridges also learn the answer format, '
         'so part of the gap to the uncompressed teachers is format adaptation.', fontsize=5.8, color=INK2, wrap=True)
for ext in ('pdf', 'png', 'svg'):
    fig.savefig(HERE / f'fig_qasper_distributions.{ext}', bbox_inches='tight', dpi=300)
(HERE / 'qasper_stats.json').write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
