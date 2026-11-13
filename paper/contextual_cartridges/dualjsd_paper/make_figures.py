"""ACL-style figures + stats for the Dual-View JSD / query-specific Cartridges report."""
import csv
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
CSV = Path('/home/da839/Query_Cart/runs/full_comparison_20261112/per_case.csv')
AUDITS = '/nfs/roberts/project/pi_mg269/shared/da839/Query_Cart/results/cartridges_ratio_syn15_dual_view_jsd_20261112_v1/dual_targets'
EVID = json.loads(Path('/home/da839/Query_Cart/runs/reranker_evidence_accuracy_20261112/summary.json').read_text())
COVER = json.loads(Path('/home/da839/Query_Cart/runs/github_dilution_coverage_20261112/summary.json').read_text())

SURFACE, INK, INK2, GRID, NEUTRAL = '#ffffff', '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.linewidth': .6, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
DS = ['longhealth', 'mtob_ke', 'qasper']
DSN = {'longhealth': 'LongHealth', 'mtob_ke': 'MTOB (KE)', 'qasper': 'QASPER'}
METRIC = {'longhealth': 'Accuracy (%) ↑', 'mtob_ke': 'chrF ↑', 'qasper': 'Reference log-PPL ↓'}
BUD = ['r05', 'r10', 'r15', 'r20']; PCT = [5, 10, 15, 20]
SIGN = {'longhealth': 1, 'mtob_ke': 1, 'qasper': -1}
SCALE = {'longhealth': 100, 'mtob_ke': 1, 'qasper': 1}
SD_AGN, SD_SPEC = 'Vanilla Cartridges (agnostic SD)', 'Specific Cartridges (specific SD)'
DJ_AGN, DJ_SPEC = 'Agnostic + DualJSD (running)', 'Specific + DualJSD (running)'
LINES = [(DJ_SPEC, 'Specific + DualJSD (ours)', C[0], 'o'), (DJ_AGN, 'Vanilla + DualJSD', C[1], 's'),
         (SD_SPEC, 'Specific + SD', C[2], 'D'), (SD_AGN, 'Vanilla Cartridges', C[3], '^'),
         ('SnapKV', 'SnapKV', C[4], 'v'), ('FastKV (attention matching)', 'FastKV (attention matching)', C[5], 'P')]
INVALID = {('mtob_ke', 'Whole KV'), ('mtob_ke', 'SnapKV'), ('mtob_ke', 'FastKV (attention matching)')}
OPTIONAL = {'FastKV (attention matching)'}

data = defaultdict(dict)  # (dataset, budget, method) -> {case: value}
for row in csv.DictReader(CSV.open()):
    data[(row['dataset'], row['budget'], row['method'])][row['case']] = float(row['value'])


def save(fig, name):
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def style(ax, ygrid=True):
    ax.spines[['top', 'right']].set_visible(False)
    if ygrid:
        ax.grid(axis='y', color=GRID, lw=.5, zorder=0)
    ax.tick_params(length=2.5, width=.6)


stats = {'paired': {}, 'main': {}, 'audit': {}}

# ---------- paired comparisons ----------
def paired(d, a, b):
    """a vs b on shared cases, summed over budgets; returns wins/losses/ties and mean delta."""
    w = l = t = 0; deltas = []; requests = set()
    for bud in BUD:
        A, B = data.get((d, bud, a), {}), data.get((d, bud, b), {})
        for case in set(A) & set(B):
            delta = SIGN[d] * (A[case] - B[case]) * SCALE[d]
            deltas.append(delta); requests.add(case)
            if delta > 1e-9: w += 1
            elif delta < -1e-9: l += 1
            else: t += 1
    return dict(wins=w, losses=l, ties=t, cells=len(deltas), requests=len(requests),
                mean_gain=st.mean(deltas) if deltas else float('nan'))


for d in DS:
    stats['paired'][d] = {'specific_vs_agnostic_SD': paired(d, SD_SPEC, SD_AGN),
                          'dualjsd_vs_SD_agnostic': paired(d, DJ_AGN, SD_AGN),
                          'dualjsd_vs_SD_specific': paired(d, DJ_SPEC, SD_SPEC),
                          'specific_dualjsd_vs_vanilla': paired(d, DJ_SPEC, SD_AGN)}

# ---------- Figure 1: method overview ----------
fig, ax = plt.subplots(figsize=(6.9, 2.05)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.0', fc=SURFACE, ec=color, lw=1.1))
    ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', fontsize=6.9, weight='bold', color=INK)
    ax.text(x + w / 2, y + h / 2 - 1.8, body, ha='center', va='center', fontsize=5.9, color=INK2, linespacing=1.25)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, color=INK2, lw=.7))
box(0.5, 9, 14, 17, 'Benchmark query', 'arrives with the\ndocument; no\nanswer label', NEUTRAL)
box(18, 9, 16, 17, 'Evidence retrieval', 'Qwen3-Reranker-0.6B\nranks 640-token chunks\n(+ joint pairs)', C[2])
box(37.5, 9, 16, 17, 'Answer-guided Q/A', 'conclusions →\nclosed-book questions;\nE answers from evidence', C[2])
box(57.5, 17.5, 18, 10, 'View E: evidence', r'$P_E(\cdot\mid$ evidence, $q$, $y_{<t})$', C[0])
box(57.5, 3, 18, 10, 'View W: whole KV', r'$P_W(\cdot\mid$ document, $q$, $y_{<t})$', C[1])
box(79.5, 9, 20, 17, 'Dual-View JSD target', r'$M_t=\frac{1}{2}(P_E+P_W)$' + '\n' + r'$w_t = 1-\mathrm{JSD}(P_E,P_W)$' + '\n' + r'loss $= w_t\,\mathrm{CE}(M_t, S_t)$', C[0])
arrow(14.8, 17.5, 17.7, 17.5); arrow(34.3, 17.5, 37.2, 17.5)
arrow(53.8, 19, 57.2, 22); arrow(53.8, 16, 57.2, 8)
arrow(75.8, 22.5, 79.2, 19); arrow(75.8, 8, 79.2, 15)
ax.text(46, 1.2, 'query-specific data (§3.1)', ha='center', fontsize=6.5, color=C[2], style='italic')
ax.text(78, 1.2, 'Dual-View JSD self-distillation into native KV (§3.2)', ha='center', fontsize=6.5, color=C[0], style='italic')
save(fig, 'fig1_overview')

# ---------- Figure 2: main results vs budget ----------
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15))
main_methods = [m for m, *_ in LINES]
for ax, d in zip(axes, DS):
    style(ax)
    entry = {}
    for bud, pct in zip(BUD, PCT):
        req = [m for m in main_methods if m not in OPTIONAL]
        shared = set.intersection(*(set(data.get((d, bud, m), {})) for m in req)) if all(
            data.get((d, bud, m)) for m in req) else set()
        entry[bud] = {'n': len(shared)}
        for m in main_methods + ['Whole KV', 'Evidence Teacher']:
            vals = [data[(d, bud, m)][c] for c in shared if c in data.get((d, bud, m), {})]
            entry[bud][m] = st.mean(vals) * SCALE[d] if vals else float('nan')
    stats['main'][d] = entry
    for m, label, color, marker in LINES:
        ys = [entry[b][m] for b in BUD]
        if (d, m) in INVALID:
            ys = [float('nan')] * len(BUD)
        ax.plot(PCT, ys, color=color, lw=1.4 if m == DJ_SPEC else 1.1, marker=marker, ms=3.6,
                label=label, zorder=4 if m == DJ_SPEC else 3, mec=SURFACE, mew=.5)
    placed = []
    lo, hi = ax.get_ylim(); span = abs(hi - lo)
    for ref, ls in (('Whole KV', (0, (4, 2))), ('Evidence Teacher', (0, (1, 1.5)))):
        y = st.mean([entry[b][ref] for b in BUD if not math.isnan(entry[b][ref])] or [float('nan')])
        if (d, ref) in INVALID or math.isnan(y):
            continue
        name = 'Whole KV' if ref == 'Whole KV' else 'Evid. ICL'
        if not (min(lo, hi) - .15 * span <= y <= max(lo, hi) + .15 * span):
            pos = (.55, .03, 'center') if d == 'mtob_ke' else (.98, .2, 'right')
            ax.text(pos[0], pos[1], f'{name}: {y:.2f} (off axis)', transform=ax.transAxes,
                    fontsize=5.8, color=INK2, ha=pos[2], va='bottom')
            continue
        ax.axhline(y, color=NEUTRAL, lw=.9, ls=ls, zorder=2)
        ty = y
        for other in placed:
            if abs(ty - other) < .07 * span:
                ty = other - .08 * span * (1 if d != 'qasper' else -1)
        placed.append(ty)
        ax.text(20.6, ty, ' ' + name, fontsize=6, color=INK2, va='center', ha='left', clip_on=False)
    ax.set_xticks(PCT); ax.set_xticklabels([f'{p}%' for p in PCT]); ax.set_xlim(4, 21)
    ax.set_xlabel('Native KV budget (% of Whole KV)'); ax.set_ylabel(METRIC[d])
    n = min(entry[b]['n'] for b in BUD)
    ax.set_title(f'{DSN[d]}  (n≥{n} per budget)', loc='left', color=INK)
    if d == 'qasper':
        ax.invert_yaxis()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.01), fontsize=6.3, columnspacing=1.4)
fig.tight_layout(w_pad=2.2)
save(fig, 'fig2_main_results')

# ---------- Figure 3: why query-specific data ----------
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0), gridspec_kw={'width_ratios': [1, 1, 1.25], 'wspace': .62})
ax = axes[0]; style(ax)
xs = range(3); w = .36
agn = [COVER[d]['arms']['vanilla']['query_term_recall'] for d in DS]
spe = [COVER[d]['arms']['specific']['query_term_recall'] for d in DS]
for j, (vals, color, label) in enumerate([(spe, C[2], 'Query-specific'), (agn, C[3], 'Agnostic')]):
    bars = ax.bar([x + (j - .5) * (w + .03) for x in xs], vals, w, color=color, label=label, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}', ha='center', fontsize=6, color=INK2)
ax.set_xticks(list(xs)); ax.set_xticklabels(['Long-\nHealth', 'MTOB', 'QASPER']); ax.set_ylim(0, 90)
ax.set_ylabel('Query-term recall (%)'); ax.set_title('(a) Practice covers the query', loc='left')
ax.legend(frameon=False, loc='upper right')
ax = axes[1]; style(ax)
ev = ['longhealth', 'qasper']
spe = [EVID[d]['fraction_specific_trajectories_with_gold'] * 100 for d in ev]
agn = [EVID[d]['fraction_agnostic_trajectories_with_gold'] * 100 for d in ev]
for j, (vals, color) in enumerate([(spe, C[2]), (agn, C[3])]):
    bars = ax.bar([x + (j - .5) * (w + .03) for x in range(2)], vals, w, color=color, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}', ha='center', fontsize=6, color=INK2)
ax.set_xticks([0, 1]); ax.set_xticklabels([DSN[d] for d in ev]); ax.set_ylim(0, 70)
ax.set_ylabel('Teacher contexts with gold (%)'); ax.set_title('(b) Teacher sees gold evidence', loc='left')
ax = axes[2]; style(ax, ygrid=False)
for i, d in enumerate(DS):
    p = stats['paired'][d]['specific_vs_agnostic_SD']; tot = max(1, p['cells'])
    left = 0
    for key, color in (('wins', C[2]), ('ties', '#d9d8d3'), ('losses', C[3])):
        frac = p[key] / tot * 100
        ax.barh(i, frac, left=left, color=color, height=.55, zorder=3, edgecolor=SURFACE, lw=1)
        if frac >= 9:
            ax.text(left + frac / 2, i, f'{p[key]}', ha='center', va='center', fontsize=6.3, color=INK)
        left += frac
ax.set_yticks(range(3)); ax.set_yticklabels([DSN[d] for d in DS]); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel('Paired cells (%): win / tie / loss')
ax.set_title('(c) Specific vs agnostic, same SD', loc='left')
save(fig, 'fig3_query_specific')

# ---------- Figure 4: Dual-View JSD ----------
audit = defaultdict(lambda: defaultdict(list))
for path in glob.glob(f'{AUDITS}/*/*/audit.json'):
    d = Path(path).parts[-3]; a = json.loads(Path(path).read_text())
    for arm, v in a['arms'].items():
        for t in v['trajectories']:
            audit[(d, arm)]['agree'].append(t['e_w_top1_agreement'] * 100)
            audit[(d, arm)]['frac'].append(t['fraction_jsd_above_0.3'] * 100)
            audit[(d, arm)]['weight'].append(t['mean_weight'])
            audit[(d, arm)]['lpe'].append(t['answer_token_mean_logprob_e'])
            audit[(d, arm)]['lpw'].append(t['answer_token_mean_logprob_w'])
stats['audit'] = {f'{d}/{arm}': {k: (st.mean(v) if v else None) for k, v in vals.items()} | {'trajectories': len(vals['agree'])}
                  for (d, arm), vals in audit.items()}
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.05), gridspec_kw={'width_ratios': [1.05, 1, 1.25], 'wspace': .62})
ax = axes[0]; style(ax)
positions, colors, series = [], [], []
for i, d in enumerate(DS):
    for j, (arm, color) in enumerate((('specific', C[0]), ('agnostic', C[1]))):
        series.append(audit[(d, arm)]['frac'] or [0]); positions.append(i + (j - .5) * .38); colors.append(color)
parts = ax.violinplot(series, positions=positions, widths=.34, showextrema=False, showmedians=True)
for body, color in zip(parts['bodies'], colors):
    body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.55)
parts['cmedians'].set_color(INK); parts['cmedians'].set_linewidth(.8)
ax.set_xticks(range(3)); ax.set_xticklabels(['Long-\nHealth', 'MTOB', 'QASPER'])
ax.set_ylabel('Tokens with JSD > 0.3 (%)'); ax.set_title('(a) E and W disagree', loc='left')
ax.plot([], [], color=C[0], lw=4, alpha=.6, label='Specific data'); ax.plot([], [], color=C[1], lw=4, alpha=.6, label='Agnostic data')
ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(.5, 1.02), fontsize=6, handlelength=1.2, columnspacing=.8)
ax.set_ylim(0, max(max(s) for s in series) * 1.22)
ax = axes[1]; style(ax)
for d, marker in zip(DS, ('o', 's', 'D')):
    lpe, lpw = audit[(d, 'specific')]['lpe'], audit[(d, 'specific')]['lpw']
    ax.scatter(lpe, lpw, s=6, marker=marker, color=C[0], alpha=.35, lw=0, label={'longhealth': 'LH', 'mtob_ke': 'MTOB', 'qasper': 'QASPER'}[d], zorder=3)
allx = [v for d in DS for v in audit[(d, 'specific')]['lpe']] or [-1]; ally = [v for d in DS for v in audit[(d, 'specific')]['lpw']] or [-1]
lo = min(min(allx), min(ally)) * 1.05; lim = [lo, 0.02]
ax.plot(lim, lim, color=NEUTRAL, lw=.7, ls=(0, (3, 2)))
ax.set_xlim(min(allx) * 1.15, 0.02); ax.set_ylim(lo, 0.02 + .2 * abs(lo))
ax.text(.03, .97, 'below diagonal: W less sure than E', transform=ax.transAxes, fontsize=5.6, color=INK2, va='top')
ax.set_xlabel('Answer log-prob under E'); ax.set_ylabel('Answer log-prob under W')
ax.set_title('(b) W rescores E\'s answers', loc='left'); ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, .9), markerscale=1.6, fontsize=5.6, ncol=3, handletextpad=.1, columnspacing=.6)
ax = axes[2]; style(ax, ygrid=False)
rows = [(d, arm) for d in DS for arm in ('specific', 'agnostic')]
for i, (d, arm) in enumerate(rows):
    p = stats['paired'][d][f'dualjsd_vs_SD_{arm}']; tot = max(1, p['cells']); left = 0
    for key, color in (('wins', C[0] if arm == 'specific' else C[1]), ('ties', '#e4e3df'), ('losses', '#8a8984')):
        frac = p[key] / tot * 100
        ax.barh(i, frac, left=left, color=color, height=.6, zorder=3, edgecolor=SURFACE, lw=1)
        if frac >= 9:
            ax.text(left + frac / 2, i, f'{p[key]}', ha='center', va='center', fontsize=6.2, color=INK)
        left += frac
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{ {'longhealth': 'LH', 'mtob_ke': 'MTOB', 'qasper': 'QASP.'}[d]} {'spec.' if arm == 'specific' else 'agn.'}" for d, arm in rows])
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Paired cells (%): win / tie / loss')
ax.set_title('(c) DualJSD vs SD, same data', loc='left')
save(fig, 'fig4_dualjsd')

(HERE / 'stats.json').write_text(json.dumps(stats, indent=2, default=str) + '\n')
print(json.dumps(stats['paired'], indent=1))
