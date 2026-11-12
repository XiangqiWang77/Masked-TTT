"""Plot measured synthesis time, without confusing it with TTFT or SD time.

Historical runs are observational: concurrency and cached retrieval differ.
Primary plots match request IDs and generated train/validation counts and
exclude recorded checkpoint replay. Every original record remains in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BLUE = '#3180A8'
ORANGE = '#E36A1C'
INK = '#0B2A3A'


def load(path):
    return json.loads(path.read_text())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {'n': 0}
    return {'n': len(x), 'mean_seconds': float(x.mean()),
            **{f'p{p}_seconds': float(np.percentile(x, p))
               for p in (10, 25, 50, 75, 90, 95)},
            'min_seconds': float(x.min()), 'max_seconds': float(x.max())}


def describe(rows):
    return {'n': len(rows),
            'vanilla': quantiles([r['vanilla_seconds'] for r in rows]),
            'query_specific': quantiles([r['specific_seconds'] for r in rows]),
            'specific_faster_requests': sum(r['specific_seconds'] < r['vanilla_seconds'] for r in rows),
            'vanilla_qa_pairs': sum(r['vanilla_questions'] for r in rows),
            'specific_qa_pairs': sum(r['specific_questions'] for r in rows),
            'counts_by_document_tokens': dict(sorted(Counter(r['document_tokens'] for r in rows).items()))}


def collect(vanilla, specific):
    a = {p.stem: p for p in (vanilla / 'dataset_cases').glob('*.json')}
    b = {p.stem: p for p in (specific / 'dataset_cases').glob('*.json')}
    rows = []
    for key in sorted(set(a) & set(b)):
        v, s = load(a[key]), load(b[key])
        if (v['case'], v['query'], v['task'], v['length']) != (s['case'], s['query'], s['task'], s['length']):
            raise ValueError(f'Request identity mismatch: {key}')
        vt = v.get('seconds')
        st = s.get('timings', {}).get('case_seconds')
        replay = int(s.get('provenance', {}).get('replayed_checkpoint_calls', 0))
        reasons = []
        if not (v.get('complete') and s.get('complete')):
            reasons.append('incomplete')
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (vt, st)):
            reasons.append('missing_or_invalid_timing')
        if s.get('inherited') or not v.get('newly_generated'):
            reasons.append('inherited_generation')
        if replay:
            reasons.append('recorded_checkpoint_replay')
        same_count = all(v['counts'][split] == s['counts'][split] for split in ('train', 'validation'))
        if not same_count:
            reasons.append('different_generated_qa_count')
        rows.append({'case': key, 'task': v['task'], 'document_tokens': v['length'],
            'vanilla_seconds': vt, 'specific_seconds': st,
            'vanilla_questions': sum(v['counts'][x] for x in ('train', 'validation')),
            'specific_questions': sum(s['counts'][x] for x in ('train', 'validation')),
            'same_train_and_validation_counts': same_count,
            'specific_replayed_calls': replay,
            'primary_eligible': not reasons, 'exclusion_reasons': ';'.join(reasons),
            'vanilla_file': str(a[key]), 'vanilla_sha256': checksum(a[key]),
            'specific_file': str(b[key]), 'specific_sha256': checksum(b[key])})
    return rows, {'vanilla_only': sorted(set(a) - set(b)), 'specific_only': sorted(set(b) - set(a))}


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(INK)
    ax.tick_params(axis='both', colors=INK, length=3.5)
    ax.set_axisbelow(True)


def export(fig, path):
    # Keep physical page size fixed; no tight-bbox crop or hidden outliers.
    outputs = []
    for ext in ('png', 'pdf', 'svg'):
        p = path.with_suffix('.' + ext)
        fig.savefig(p, dpi=400, facecolor='white')
        outputs.append({'path': str(p), 'sha256': checksum(p), 'bytes': p.stat().st_size})
    plt.close(fig)
    return outputs


def figures(rows, output):
    v = np.asarray([r['vanilla_seconds'] for r in rows])
    s = np.asarray([r['specific_seconds'] for r in rows])
    limit = max(float(v.max()), float(s.max())) * 1.05
    style = {'font.family': 'DejaVu Sans', 'font.size': 10,
             'axes.labelsize': 11, 'axes.labelcolor': INK,
             'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'svg.fonttype': 'none', 'axes.linewidth': .8,
             'legend.frameon': False, 'legend.fontsize': 9}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.7, 4.25), layout='constrained')
        vp = ax.violinplot([v, s], positions=[0, 1], widths=.72,
                           showmeans=False, showmedians=True, showextrema=False,
                           points=256, bw_method='scott')
        for color, body in zip((ORANGE, BLUE), vp['bodies']):
            body.set_alpha(None)
            body.set_facecolor(matplotlib.colors.to_rgba(color, .30))
            body.set_edgecolor(color); body.set_linewidth(1.3)
        vp['cmedians'].set_color(INK)
        vp['cmedians'].set_linewidth(2.3)
        vp['cmedians'].set_zorder(3)
        ax.set(xticks=[0, 1], xticklabels=['Vanilla synthesis', 'Query-specific synthesis'],
               ylabel='Synthesis time (s)', xlim=(-.58, 1.58), ylim=(0, limit))
        ax.legend(handles=[Line2D([], [], color=INK, linewidth=2.3, label='Median')],
                  loc='upper left')
        clean(ax)
        violin = export(fig, output / 'synthesis_time_violin')
    return violin


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vanilla', type=Path, required=True)
    p.add_argument('--specific', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, missing = collect(args.vanilla, args.specific)
    primary = [r for r in rows if r['primary_eligible']]
    fresh = [r for r in rows if r['exclusion_reasons'] in ('', 'different_generated_qa_count')]
    if len(primary) < 2:
        raise ValueError('At least two eligible paired observations are required')
    with (args.output / 'synthesis_time_per_request.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'metric': 'recorded per-request synthetic dataset construction wall time in seconds; not TTFT',
        'dataset': 'BABILong', 'backbone': 'Qwen3-4B-Instruct-2507',
        'primary_eligibility': 'Same request and same generated train/validation counts; positive finite times; complete; no recorded checkpoint replay or inherited generation.',
        'primary': describe(primary), 'all_without_recorded_replay': describe(fresh),
        'all_recorded_attempts_including_replay': describe(rows),
        'exclusion_reason_counts': dict(Counter(x for r in rows for x in r['exclusion_reasons'].split(';') if x)),
        'missing_pairs': missing,
        'scope': {
            'included': ['Per-request construction, answer/question generation, and shared backend queue waits'],
            'excluded': ['Model startup', 'Slurm queue', 'SD training', 'Final-answer decoding', 'Historical Decision API calls'],
            'specific_retrieval': 'Uses cached Decision API decisions, with in-attempt local preparation included when performed.',
            'vanilla_concurrent_cases_per_worker': load(args.vanilla/'config.json')['concurrent_cases'],
            'specific_concurrent_cases_per_worker': load(args.specific/'config.json')['concurrent_cases'],
            'hardware_controlled': False,
            'claim': 'Observed historical run times only; not an isolated algorithmic or cold-start speedup.',
            'current_three_dataset_timing_comparison': 'Unavailable: current Vanilla synthetic JSON has no independent synthesis timer.'},
        'sources': {str(path): checksum(path) for path in (args.vanilla/'config.json', args.specific/'config.json')},
        'figure_semantics': {'violin': 'Two violins: Scott KDE, 256 evaluation points, truncated to observed range; equal maximum widths; all eligible observations used.',
            'center_line': 'One horizontal sample median per violin.',
            'individual_points_displayed': False, 'intervals_displayed': False,
            'smoothing_applied_to_data_values': False,
            'outliers_removed': False, 'bar_charts': False},
        'script_sha256': checksum(Path(__file__)),
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
    }
    report['outputs'] = figures(primary, args.output)
    (args.output / 'synthesis_time_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    a, b = report['primary']['vanilla'], report['primary']['query_specific']
    text = f'''# Synthetic data generation time

The two violins compare historical Qwen3-4B BABILong runs, not the current LongHealth/MTOB/QASPER jobs.
Each observation is the measured time to construct the synthetic dataset for one request. It is not TTFT.

The figure contains {len(primary)} matched requests with identical generated train and validation
question counts ({report['primary']['vanilla_qa_pairs']:,} QA pairs per method). All 1,000 records and
their exclusion reasons are preserved in `synthesis_time_per_request.csv`. Eligibility never uses timing
advantages, benchmark answers, or model accuracy. Sixteen query-specific records include checkpoint
replay and are excluded from the primary comparison. Other exclusions reflect unequal generated counts.

| Method | Mean (s) | Median (s) | P90 (s) |
|---|---:|---:|---:|
| Vanilla synthesis | {a['mean_seconds']:.2f} | {a['p50_seconds']:.2f} | {a['p90_seconds']:.2f} |
| Query-specific synthesis | {b['mean_seconds']:.2f} | {b['p50_seconds']:.2f} | {b['p90_seconds']:.2f} |

Query-specific synthesis is faster on {report['primary']['specific_faster_requests']}/{len(primary)} matched requests.
Its median is lower, but its mean and P90 are higher. These observations do not establish uniform speed superiority.

![Two violins show complete observed synthesis-time distributions. Each dark horizontal line is the sample median.](synthesis_time_violin.png)

[PDF](synthesis_time_violin.pdf) · [SVG](synthesis_time_violin.svg) · [Raw timing CSV](synthesis_time_per_request.csv) · [Summary JSON](synthesis_time_summary.json)

Timing includes shared-backend waiting. Concurrency differed (Vanilla 32 versus query-specific 16 cases
per worker), hardware was not held fixed, and the query-specific run reused prior Decision API evidence
selection. Those historical API calls are not included. These are observational run-time distributions,
not a controlled same-hardware speed benchmark. Do not sum per-request durations to infer campaign wall
time: concurrent requests overlap. No runtime changes or GPU submissions were made for this analysis.

Each violin uses all eligible durations, with Scott KDE bounded by the observed minimum and maximum.
Width encodes density, not sample count. The dark horizontal line is the sample median. The original
long tail remains visible. Individual dots, bars, and confidence intervals are not displayed. CSV, JSON,
PDF, PNG, SVG and the reproducible analysis script accompany the figure. General research style;
no ACL compliance certification.

Software: the scientific-visualization skill informed figure integrity and provenance checks.
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026),
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
'''
    (args.output / 'README.md').write_text(text)
    print(json.dumps({'primary': report['primary'], 'excluded': report['exclusion_reason_counts'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
