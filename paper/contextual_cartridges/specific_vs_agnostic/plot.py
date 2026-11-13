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


fig = plt.figure(figsize=(6.9, 4.9))
gs = fig.add_gridspec(2, 2, hspace=.55, wspace=.32)

# (a) two donuts: query relevance of practice questions
cats = [('strong', 'shares ≥ half of the query\'s content terms', BLUE),
        ('partial', 'shares some query terms', '#86b6ef'),
        ('none', 'unrelated to the query', '#d6d5d0')]
q['cat'] = pd.cut(q.recall, [-1, 0, .4999, 1.01], labels=['none', 'partial', 'strong'])
sub = gs[0, 0].subgridspec(1, 2, wspace=.05)
for j, (arm, name) in enumerate((('agnostic', 'query-agnostic'), ('specific', 'query-specific'))):
    ax = fig.add_subplot(sub[j])
    share = q[q.arm == arm].cat.value_counts(normalize=True)
    vals = [share.get(c, 0) for c, *_ in cats]
    stats[f'relevance/{arm}'] = dict(zip([c for c, *_ in cats], map(float, vals)))
    ax.pie(vals, colors=[c for *_, c in cats], startangle=90, counterclock=False,
           wedgeprops=dict(width=.42, edgecolor=SURFACE, linewidth=1.2))
    ax.text(0, 0, f'{vals[0]*100:.0f}%\nstrong', ha='center', va='center', fontsize=8, color=INK)
    ax.set_title(name, fontsize=7.5, color=ORANGE if arm == 'agnostic' else BLUE)
    if j == 0:
        ax.text(-1.3, 1.75, '(a) Query relevance of the practice questions', fontsize=8, color=INK)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for *_, c in cats], labels=[l for _, l, _ in cats],
           loc='upper left', bbox_to_anchor=(.08, .565), frameon=False, fontsize=6.3, ncol=1)

# (b) token level: share of training-answer words that are query terms
ax = fig.add_subplot(gs[0, 1]); clean(ax)
for i, d in enumerate(DS):
    a = q[(q.dataset == d) & (q.arm == 'agnostic')].answer_query_density * 100
    b = q[(q.dataset == d) & (q.arm == 'specific')].answer_query_density * 100
    split_violin(ax, a, b, i, bw=.3)
    ax.text(i, max(a.max(), b.max()) + 2, f'×{b.mean() / max(a.mean(), 1e-3):.0f}' if a.mean() > 0.05 else f'{b.mean():.0f}% vs {a.mean():.1f}%',
            ha='center', fontsize=6.6, color=INK2)
    stats[f'answer_density/{d}'] = dict(agnostic=float(a.mean()), specific=float(b.mean()))
ax.set_xticks(range(3)); ax.set_xticklabels([DSN[d] for d in DS])
ax.set_ylabel('query terms / answer words (%)')
ax.set_title('(b) Query terms in the training answers', loc='left', fontsize=8)
ax.set_xlabel('violin left half: query-agnostic · right half: query-specific', fontsize=6.6, color=INK2)
ax.set_xlim(-.6, 2.6)

# (c) teacher contexts that contain gold evidence
ax = fig.add_subplot(gs[1, 0]); clean(ax)
for i, d in enumerate(['longhealth', 'qasper']):
    e = ev[ev.dataset == d]
    a = e.agnostic_with_gold / e.agnostic_trajectories * 100
    b = e.specific_with_gold / e.specific_trajectories * 100
    for v, off, color in ((a, -.13, ORANGE), (b, .13, BLUE)):
        ax.scatter(i + off + rng.uniform(-.06, .06, len(v)), v, s=10, color=color, alpha=.7, lw=0, zorder=3)
        ax.plot([i + off - .1, i + off + .1], [v.mean()] * 2, color=color, lw=2, zorder=4)
        ax.text(i + off, -9, f'{v.mean():.0f}%', ha='center', fontsize=6.4, color=color)
    stats[f'gold/{d}'] = dict(agnostic=float(a.mean()), specific=float(b.mean()), n=len(e))
ax.set_xticks([0, 1]); ax.set_xticklabels(['LongHealth', 'QASPER']); ax.set_ylim(-14, 105)
ax.set_ylabel('teacher contexts with gold evidence (%)')
ax.set_title('(c) Teacher contexts containing gold evidence', loc='left', fontsize=8)
ax.scatter([], [], color=ORANGE, label='query-agnostic'); ax.scatter([], [], color=BLUE, label='query-specific')
ax.legend(frameon=False, fontsize=6.4, loc='upper right')

# (d) paired outcome under identical SD
ax = fig.add_subplot(gs[1, 1]); clean(ax, grid='x')
sign = {'longhealth': 100, 'mtob_ke': 1, 'qasper': -1}
unit = {'longhealth': 'acc. pts', 'mtob_ke': 'chrF', 'qasper': '−log-PPL'}
for i, d in enumerate(DS[::-1]):
    g = o[o.dataset == d].assign(gain=lambda x: sign[d] * (x.specific - x.agnostic)).groupby('case').gain.mean()
    z = g / np.abs(g).max()
    k = gaussian_kde(z, bw_method=.4); grid = np.linspace(-1.05, 1.05, 200); dens = k(grid); dens = dens / dens.max() * .38
    ax.fill_between(grid, i + .05, i + .05 + dens, color=BLUE, alpha=.3, lw=0); ax.plot(grid, i + .05 + dens, color=BLUE, lw=.9)
    ax.scatter(z, i - .12 + rng.uniform(-.06, .06, len(z)), s=10, color=BLUE, alpha=.8, lw=0, zorder=3)
    win, lose = (g > 1e-9).mean(), (g < -1e-9).mean()
    ax.text(1.06, i, f'{DSN[d]}\nmean {g.mean():+.2f} {unit[d]}\nbetter {win*100:.0f}% · worse {lose*100:.0f}%',
            fontsize=6.1, color=INK2, va='center', transform=ax.get_yaxis_transform())
    stats[f'outcome/{d}'] = dict(mean=float(g.mean()), win=float(win), lose=float(lose), n=len(g))
ax.axvline(0, color=INK2, lw=.8, ls=(0, (3, 2)))
ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(-1.1, 1.1)
ax.set_xlabel('specific − agnostic (scaled per dataset; avg. over 4 budgets)')
ax.text(.01, .985, '← agnostic better', transform=ax.transAxes, fontsize=6, color=INK2, va='top')
ax.text(.99, .985, 'specific better →', transform=ax.transAxes, fontsize=6, color=INK2, va='top', ha='right')
ax.set_title('(d) Paired gain per request, same SD', loc='left', fontsize=8)
fig.text(.07, 1.0, 'Why query-specific practice data beats query-agnostic data (Qwen3-4B, same SD, 15 requests per dataset)',
         fontsize=8.6, color=INK)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(HERE / f'specific_vs_agnostic.{ext}', bbox_inches='tight', dpi=300)
(HERE / 'stats.json').write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=0))
