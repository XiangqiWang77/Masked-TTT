"""Single panel: LongHealth Hit@k of gold evidence (no title; legend + axes only)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
d = json.loads((OUT / 'summary.json').read_text())['longhealth']
SURFACE, INK2, GRID = '#fcfcfb', '#52514e', '#e4e3df'
BLUE, ORANGE, NEUTRAL = '#2a78d6', '#eb6834', '#8a8984'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ks = ['1', '3', '5']; xs = range(len(ks)); w = .34
for j, (method, color, label) in enumerate([('reranker', BLUE, 'Qwen3-Reranker-0.6B'),
                                            ('lexical', ORANGE, 'Lexical term overlap')]):
    values = [d['hit_at_k'][method][k] * 100 for k in ks]
    bars = ax.bar([x + (j - .5) * (w + .02) for x in xs], values, w, color=color, label=label,
                  zorder=3, edgecolor=SURFACE, linewidth=1)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}', ha='center', fontsize=8.5, color=INK2)
for x, k in zip(xs, ks):
    r = d['hit_at_k']['random'][k] * 100
    ax.plot([x - .4, x + .4], [r, r], color=NEUTRAL, lw=1.5, ls=(0, (3, 2)), zorder=4,
            label='Random chunk order' if x == 0 else None)
ax.set_xticks(list(xs)); ax.set_xticklabels([f'Hit@{k}' for k in ks])
ax.set_xlabel('Top-k reranked chunks')
ax.set_ylabel('Requests with gold evidence in top k (%)')
ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis='y', color=GRID, lw=.6, zorder=0); ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'reranker_hit_at_k.{ext}', dpi=220, bbox_inches='tight', facecolor=SURFACE)
print('saved')
