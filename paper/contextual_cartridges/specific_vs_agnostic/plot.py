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
BLUE, ORANGE = '#2a78d6', '#eb6834'
DS = ['longhealth', 'mtob_ke', 'qasper']
DSN = {'longhealth': 'LongHealth', 'mtob_ke': 'MTOB', 'qasper': 'QASPER'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                     'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 8, 'axes.linewidth': .7})
rng = np.random.default_rng(0)
q = pd.read_csv(HERE / 'questions.csv')
o = pd.read_csv(HERE / 'outcome.csv')
ev = pd.read_csv('/home/da839/Query_Cart/runs/reranker_evidence_accuracy_20261112/per_request.csv')
stats = {}


def clean(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        getattr(ax, f'{grid}axis').grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)


def split_violin(ax, a, b, pos, width=.36, bw=.35):
    for v, side, color in ((a, -1, ORANGE), (b, 1, BLUE)):
        v = np.asarray(v, float)
        grid = np.linspace(v.min(), v.max(), 200)
        d = gaussian_kde(v, bw_method=bw)(grid) if np.ptp(v) > 0 else np.ones_like(grid)
        d = d / d.max() * width
        ax.fill_betweenx(grid, pos, pos + side * d, color=color, alpha=.35, lw=0)
        ax.plot(pos + side * d, grid, color=color, lw=.9)
        ax.plot([pos, pos + side * width * .9], [np.median(v)] * 2, color=color, lw=1.6)


fig = plt.figure(figsize=(7.0, 2.85))
gs = fig.add_gridspec(1, 4, width_ratios=[.62, 1.15, .62, 1.35], wspace=.5)

# (a) stacked donuts: query relevance of practice questions
cats = [('strong', 'related (≥ half of query terms)', BLUE),
        ('partial', 'partly related', '#86b6ef'),
        ('none', 'unrelated to the query', '#d6d5d0')]
q['cat'] = pd.cut(q.recall, [-1, 0, .4999, 1.01], labels=['none', 'partial', 'strong'])
sub = gs[0].subgridspec(2, 1, hspace=.55)
for j, (arm, name, color) in enumerate((('agnostic', 'query-agnostic', ORANGE), ('specific', 'query-specific', BLUE))):
    ax = fig.add_subplot(sub[j])
    share = q[q.arm == arm].cat.value_counts(normalize=True)
    vals = [share.get(c, 0) for c, *_ in cats]
    stats[f'relevance/{arm}'] = dict(zip([c for c, *_ in cats], map(float, vals)))
    ax.pie(vals, colors=[c for *_, c in cats], startangle=90, counterclock=False,
           wedgeprops=dict(width=.4, edgecolor=SURFACE, linewidth=1.0))
    ax.text(0, 0, f'{vals[0]*100:.0f}%', ha='center', va='center', fontsize=8.5, color=INK)
    ax.text(0, -1.3, name + f'\n{vals[2]*100:.0f}% unrelated', ha='center', va='top', fontsize=6.2, color=color)
    if j == 0:
        ax.set_title('(a) Practice questions\n     related to the query', loc='left', fontsize=7.8, x=-.25)

# (b) token level
ax = fig.add_subplot(gs[1]); clean(ax)
for i, d in enumerate(DS):
    a = q[(q.dataset == d) & (q.arm == 'agnostic')].answer_query_density * 100
    b = q[(q.dataset == d) & (q.arm == 'specific')].answer_query_density * 100
    split_violin(ax, a, b, i, bw=.3)
    ax.text(i + .2, b.quantile(.97) + 3, f'×{b.mean() / max(a.mean(), 1e-3):.0f}', ha='center', fontsize=7, color=BLUE)
    stats[f'answer_density/{d}'] = dict(agnostic=float(a.mean()), specific=float(b.mean()))
ax.set_xticks(range(3)); ax.set_xticklabels(['LongH.', 'MTOB', 'QASPER']); ax.set_xlim(-.55, 2.55)
ax.set_ylim(-2, 75)
ax.set_ylabel('query terms in answer (%)')
ax.set_title('(b) Training answers focus on\n     the query (per answer)', loc='left', fontsize=7.8)

# (c) LongHealth: teacher contexts with gold evidence
ax = fig.add_subplot(gs[2]); clean(ax)
e = ev[ev.dataset == 'longhealth']
for i, (col_w, col_n, color, name) in enumerate((('agnostic_with_gold', 'agnostic_trajectories', ORANGE, 'agn.'),
                                                 ('specific_with_gold', 'specific_trajectories', BLUE, 'spec.'))):
    v = e[col_w] / e[col_n] * 100
    ax.scatter(i + rng.uniform(-.14, .14, len(v)), v, s=9, color=color, alpha=.7, lw=0, zorder=3)
    ax.plot([i - .22, i + .22], [v.mean()] * 2, color=color, lw=2.2, zorder=4)
    ax.text(i, v.mean() + 4, f'{v.mean():.0f}%', ha='center', fontsize=7, color=color, zorder=5,
            bbox=dict(fc=SURFACE, ec='none', pad=.5, alpha=.85))
    stats[f'gold/longhealth/{name}'] = float(v.mean())
ax.set_xticks([0, 1]); ax.set_xticklabels(['agn.', 'spec.'], fontsize=7); ax.set_xlim(-.6, 1.6); ax.set_ylim(-3, 103)
ax.set_ylabel('teacher contexts with gold (%)')
ax.set_title('(c) Teacher sees the\n     evidence (LongHealth)', loc='left', fontsize=7.8)

# (d) outcome
ax = fig.add_subplot(gs[3]); clean(ax, grid='x')
sign = {'longhealth': 100, 'mtob_ke': 1, 'qasper': -1}
unit = {'longhealth': 'acc. pts', 'mtob_ke': 'chrF', 'qasper': 'log-PPL ↓'}
for i, d in enumerate(DS[::-1]):
    g = o[o.dataset == d].assign(gain=lambda x: sign[d] * (x.specific - x.agnostic)).groupby('case').gain.mean()
    z = g / np.abs(g).max()
    grid = np.linspace(-1.05, 1.05, 200); dens = gaussian_kde(z, bw_method=.4)(grid); dens = dens / dens.max() * .36
    ax.fill_between(grid, i + .04, i + .04 + dens, color=BLUE, alpha=.3, lw=0); ax.plot(grid, i + .04 + dens, color=BLUE, lw=.9)
    ax.scatter(z, i - .12 + rng.uniform(-.06, .06, len(z)), s=8, color=BLUE, alpha=.8, lw=0, zorder=3)
    win, lose = (g > 1e-9).mean(), (g < -1e-9).mean()
    shown = g.mean() if d != 'qasper' else -g.mean()
    ax.text(-1.08, i + .5, f'{DSN[d]}: {shown:+.2f} {unit[d]} · better {win*100:.0f}% / worse {lose*100:.0f}%',
            fontsize=5.9, color=INK2)
    stats[f'outcome/{d}'] = dict(mean=float(g.mean()), win=float(win), lose=float(lose), n=len(g))
ax.axvline(0, color=INK2, lw=.8, ls=(0, (3, 2)))
ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-.35, 2.78)
ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(['← agnostic better', '0', 'specific better →'], fontsize=6.3)
ax.set_xlabel('paired gain per request (scaled per dataset)', fontsize=6.6)
ax.set_title('(d) Result: specific vs agnostic,\n     same SD', loc='left', fontsize=7.8)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for *_, c in cats] +
                   [plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=.6), plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=.6)],
           labels=[l for _, l, _ in cats] + ['query-agnostic data', 'query-specific data'],
           loc='upper center', bbox_to_anchor=(.5, -.06), ncol=5, frameon=False, fontsize=6.5, handlelength=1.1)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(HERE / f'specific_vs_agnostic.{ext}', bbox_inches='tight', dpi=300)
(HERE / 'stats.json').write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=0))
