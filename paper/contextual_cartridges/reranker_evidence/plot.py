"""Figure: what the Qwen3-Reranker-0.6B evidence extraction does and how accurate it is."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
S = json.loads((OUT / 'summary.json').read_text())
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, VIOLET, NEUTRAL = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#8a8984'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.titlecolor': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})

fig = plt.figure(figsize=(12, 7.2))
grid = fig.add_gridspec(2, 3, height_ratios=[.62, 1.35], hspace=.42, wspace=.32)

# A. pipeline
ax = fig.add_subplot(grid[0, :]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
steps = [('Document', 'split into\n640-token chunks\n(char spans kept)'),
         ('Lexical prefilter', 'query-term overlap\nkeeps ≤64 chunks\n(+4 scaffold)'),
         ('Qwen3-Reranker\n0.6B', 'yes/no per chunk\nscore =\nP(yes | query, chunk)'),
         ('Pair scoring', 'top-16 chunks →\n32 entity/adjacent\npairs, joint yes/no'),
         ('Evidence groups', '≤32 groups:\ntop pairs interleaved\nwith singletons'),
         ('Teacher context', 'each synthetic Q/A\nis bound to one\nevidence group')]
width, gap = 14.2, 2.6
for i, (title, body) in enumerate(steps):
    x = 1 + i * (width + gap)
    accent = BLUE if 'Reranker' in title or 'Pair' in title else NEUTRAL
    ax.add_patch(FancyBboxPatch((x, 4), width, 21, boxstyle='round,pad=0.3,rounding_size=1.2',
                                fc=SURFACE, ec=accent, lw=1.6 if accent == BLUE else 1))
    ax.text(x + width / 2, 20.5, title, ha='center', va='center', fontsize=8.6, linespacing=1.1, weight='bold', color=INK)
    ax.text(x + width / 2, 10.5, body, ha='center', va='center', fontsize=7.4, color=INK2, linespacing=1.35)
    if i < len(steps) - 1:
        ax.annotate('', xy=(x + width + gap - .3, 14.5), xytext=(x + width + .5, 14.5),
                    arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
ax.text(0, 29.5, 'A. How evidence is extracted (no benchmark answers, no generation)', fontsize=10.5,
        weight='bold', color=INK, va='top')

# B. Hit@k per dataset
names = {'longhealth': 'LongHealth', 'qasper': 'QASPER'}
for col, dataset in enumerate(('longhealth', 'qasper')):
    ax = fig.add_subplot(grid[1, col]); d = S[dataset]
    ks = ['1', '3', '5']; xs = range(len(ks)); w = .34
    for j, (method, color, label) in enumerate([('reranker', BLUE, 'Qwen3-Reranker-0.6B'),
                                                ('lexical', ORANGE, 'Lexical term overlap')]):
        values = [d['hit_at_k'][method][k] * 100 for k in ks]
        bars = ax.bar([x + (j - .5) * (w + .02) for x in xs], values, w, color=color, label=label,
                      zorder=3, edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}', ha='center', fontsize=7.6, color=INK2)
    for x, k in zip(xs, ks):
        r = d['hit_at_k']['random'][k] * 100
        ax.plot([x - .4, x + .4], [r, r], color=NEUTRAL, lw=1.4, ls=(0, (3, 2)), zorder=4,
                label='Random chunk order' if x == 0 else None)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f'Hit@{k}' for k in ks])
    ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='y', color=GRID, lw=.6, zorder=0); ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('% requests with a gold chunk in top k' if col == 0 else '')
    ax.set_title(f"{'B' if col == 0 else 'C'}. {names[dataset]}: is gold evidence ranked first?\n"
                 f"n={d['requests']} requests, {d['mean_chunks']:.0f} chunks/doc, MRR {d['mrr']['reranker']:.2f} vs {d['mrr']['lexical']:.2f}",
                 fontsize=9.5, loc='left')
    if col == 0:
        ax.legend(frameon=False, fontsize=7.8, loc='upper left')

# D. what the Teacher actually saw
ax = fig.add_subplot(grid[1, 2])
datasets = ['longhealth', 'qasper']; w = .34
spec = [S[d]['fraction_specific_trajectories_with_gold'] * 100 for d in datasets]
agn = [S[d]['fraction_agnostic_trajectories_with_gold'] * 100 for d in datasets]
for j, (values, color, label) in enumerate([(spec, AQUA, 'Query-specific: reranked evidence group'),
                                            (agn, VIOLET, 'Agnostic: self-study chunk')]):
    bars = ax.bar([i + (j - .5) * (w + .02) for i in range(2)], values, w, color=color, label=label,
                  zorder=3, edgecolor=SURFACE, linewidth=1)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}%', ha='center', fontsize=7.8, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels([names[d] for d in datasets])
ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis='y', color=GRID, lw=.6, zorder=0); ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('% of 15 training trajectories')
ax.set_title('D. Does the Teacher context contain gold evidence?\n(15 trajectories per request and arm)',
             fontsize=9.5, loc='left')
ax.legend(frameon=False, fontsize=7.8, loc='upper right')

fig.text(.01, .01, 'Gold = LongHealth answer_location / QASPER annotated evidence paragraphs, used only for this audit. '
         'MTOB has no evidence labels. Match = central 60-character snippet of a gold span.',
         fontsize=7.2, color=INK2)
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'reranker_evidence_accuracy.{ext}', dpi=200, bbox_inches='tight', facecolor=SURFACE)
print('saved')
