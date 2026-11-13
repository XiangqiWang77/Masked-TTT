"""Four violins: Qwen3-4B BABILong (historical) and OLMo-3-7B LongHealth synthesis time.

Reads only the two published CSVs; no GPU or private data. Writes
synthesis_time_violin.{png,pdf,svg} and the `olmo3` block of the summary JSON.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_synthesis_time_distribution import BLUE, INK, ORANGE, clean, export, quantiles

root = Path(__file__).resolve().parent


def eligible(name, keys):
    with (root / name).open() as f:
        rows = [r for r in csv.DictReader(f) if r['primary_eligible'] == 'True']
    return rows, [np.asarray([float(r[k]) for r in rows]) for k in keys]


qwen_rows, (qv, qs) = eligible('synthesis_time_per_request.csv', ('vanilla_seconds', 'specific_seconds'))
olmo_rows, (ov, os_) = eligible('olmo3_synthesis_time_per_request.csv', ('vanilla_seconds', 'specific_seconds'))
data, pos = [qv, qs, ov, os_], [0, 1, 2.6, 3.6]
style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 11, 'axes.labelcolor': INK,
         'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
         'axes.linewidth': .8, 'legend.frameon': False, 'legend.fontsize': 9}
with plt.rc_context(style):
    fig, ax = plt.subplots(figsize=(7.4, 4.4), layout='constrained')
    vp = ax.violinplot(data, positions=pos, widths=.72, showmeans=False, showmedians=True,
                       showextrema=False, points=256, bw_method='scott')
    for color, body in zip((ORANGE, BLUE, ORANGE, BLUE), vp['bodies']):
        body.set_alpha(None)
        body.set_facecolor(matplotlib.colors.to_rgba(color, .30))
        body.set_edgecolor(color); body.set_linewidth(1.3)
    vp['cmedians'].set_color(INK); vp['cmedians'].set_linewidth(2.3); vp['cmedians'].set_zorder(3)
    limit = max(float(x.max()) for x in data) * 1.05
    ax.set(xticks=[.5, 3.1], ylabel='Synthesis time per request (s)', xlim=(-.6, 4.2), ylim=(0, limit))
    ax.set_xticklabels([f'Qwen3-4B · BABILong\n(n = {len(qwen_rows)}, matched QA counts)',
                        f'OLMo-3-7B · LongHealth\n(n = {len(olmo_rows)}, 15 QA / arm)'])
    ax.axvline(1.8, color=INK, linewidth=.6, alpha=.25)
    ax.legend(handles=[Patch(facecolor=matplotlib.colors.to_rgba(ORANGE, .30), edgecolor=ORANGE, label='Vanilla synthesis'),
                       Patch(facecolor=matplotlib.colors.to_rgba(BLUE, .30), edgecolor=BLUE, label='Query-specific synthesis'),
                       Line2D([], [], color=INK, linewidth=2.3, label='Median')], loc='upper right')
    clean(ax)
    outputs = export(fig, root / 'synthesis_time_violin')

summary_path = root / 'synthesis_time_summary.json'
summary = json.loads(summary_path.read_text())
summary['olmo3'] = {
    'backbone': 'OLMo-3-7B-Instruct', 'dataset': 'LongHealth', 'hardware': 'NVIDIA H200 (gpu_devel), vLLM',
    'campaign': 'olmo3_full_table_20261113', 'n': len(olmo_rows),
    'in_175_request_table': sum(r['in_175_request_table'] == 'True' for r in olmo_rows),
    'questions_per_arm': 15,
    'vanilla': quantiles(ov), 'query_specific': quantiles(os_),
    'specific_faster_requests': int(sum(s < v for v, s in zip(ov, os_))),
    'timing_method': 'see olmo_synthesis_time.py docstring (segment start and per-arm target-file mtimes)',
}
summary['figure_semantics'] = dict(summary.get('figure_semantics', {}),
    violin='Four violins: Qwen3-4B BABILong pair and OLMo-3-7B LongHealth pair; Scott KDE, 256 points, truncated to observed range; equal maximum widths.')
summary['figure_outputs'] = outputs
summary_path.write_text(json.dumps(summary, indent=2) + '\n')
print(f'Qwen n={len(qwen_rows)}, OLMo n={len(olmo_rows)}')
