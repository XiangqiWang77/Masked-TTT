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
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'
DSN = {'longhealth': 'LongHealth', 'mtob_ke': 'MTOB', 'qasper': 'QASPER'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                     'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 8, 'axes.linewidth': .7})
snap = json.loads((HERE / 'snapshot.json').read_text())
req = pd.DataFrame(snap['requests']); tok = pd.DataFrame(snap['tokens']); fate = snap['fate']

fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6), gridspec_kw={'width_ratios': [1.05, 1, 1], 'wspace': .38})

# (a) pie: fate of every proposed query-specific question
ax = axes[0]
order = [('accepted (teacher answer complete)', 'accepted', BLUE),
         ('rejected: not closed-book', 'rejected: not closed-book (≈)', ORANGE),
         ('teacher answer incomplete / over quota', 'teacher answer incomplete or request full', YELLOW),
         ('malformed', 'malformed JSON', MAGENTA),
         ('duplicate question', 'duplicate', AQUA)]
vals = [fate.get(k, 0) for k, _, _ in order]
wedges, _ = ax.pie(vals, colors=[c for *_, c in order], startangle=90, counterclock=False,
                   wedgeprops=dict(width=.42, edgecolor=SURFACE, linewidth=1.2))
tot = sum(vals)
ax.legend(wedges, [f"{lab.replace(chr(10), ' ')}  {v/tot*100:.0f}% ({v:,})" for v, (_, lab, _) in zip(vals, order)],
          frameon=False, fontsize=5.8, loc='upper center', bbox_to_anchor=(.5, .02), handlelength=1.0)
ax.text(0, 0, f'{tot:,}\nproposed\nquestions', ha='center', va='center', fontsize=7, color=INK)
ax.set_title('(a) What happens to a proposed question', loc='left', y=1.08)

# (b) token-length distributions
ax = axes[1]
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
grid = np.logspace(np.log10(5), np.log10(1100), 300)
for arm, color in (('specific', BLUE), ('agnostic', ORANGE)):
    for col, ls in (('question', (0, (3, 1.5))), ('answer', '-')):
        v = tok[tok.arm == arm][col].clip(lower=5).values
        d = gaussian_kde(np.log10(v), bw_method=.18)(np.log10(grid))
        ax.plot(grid, d, color=color, ls=ls, lw=1.4,
                label=f'{arm} {col} (median {int(np.median(v))})')
        if col == 'answer':
            ax.fill_between(grid, 0, d, color=color, alpha=.12, lw=0)
ax.set_xscale('log'); ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(['10', '30', '100', '300', '1000'])
ax.set_yticks([]); ax.set_xlabel('tokens per trajectory (log scale)')
ax.legend(frameon=False, fontsize=5.6, loc='upper left', handlelength=1.8)
ax.set_title(f'(b) Token lengths ({len(tok):,} trajectories)', loc='left', y=1.08)

# (c) accepted questions per request
ax = axes[2]
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
done = req[req.status.isin(['complete', 'shortfall'])]
bins = np.arange(0, 17)
bottom = np.zeros(16)
for d, color in (('longhealth', BLUE), ('qasper', AQUA), ('mtob_ke', ORANGE)):
    c = np.bincount(done[done.dataset == d].accepted.clip(upper=15), minlength=16)[:16]
    ax.bar(bins[:-1], c, .82, bottom=bottom, color=color, label=DSN[d], zorder=3, edgecolor=SURFACE, lw=.4)
    bottom += c
ax.set_yscale('symlog', linthresh=5); ax.set_yticks([0, 2, 5, 10, 50, 200]); ax.set_yticklabels(['0', '2', '5', '10', '50', '200'])
ax.set_xticks([0, 5, 10, 15]); ax.set_xlabel('accepted query-specific questions per request (target 15)')
ax.set_ylabel('requests')
comp = req[req.status == 'complete']
ax.text(.02, .97, f"{(done.accepted >= 15).sum()}/{len(done)} requests reached 15\n"
        f"median attempts: {int(comp.attempts.median())} (cap 128)", transform=ax.transAxes, va='top', fontsize=6.2, color=INK2)
ax.legend(frameon=False, fontsize=6.2, loc='center left', bbox_to_anchor=(0, .55))
ax.set_title('(c) Questions per request', loc='left', y=1.08)
fig.text(.07, 1.1, f'OLMo-3-7B per-question synthesis, snapshot of the running job ({len(done)} of 300 requests finished)',
         fontsize=8.2, color=INK)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(HERE / f'olmo_synthesis_stats.{ext}', bbox_inches='tight', dpi=300)
print(done.groupby('dataset').accepted.describe().round(1))
print(tok.groupby('arm')[['question', 'answer']].median())
print(comp.groupby('dataset').attempts.median())
