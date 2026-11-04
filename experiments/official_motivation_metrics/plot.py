#!/usr/bin/env python3
"""Render a three-facet figure from completed attention records and gold coverage."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT/'docs/analysis/olmo-motivations-20261104/data_snapshot.csv'
LENGTHS = (8192, 16384, 32768, 65536)
METHODS = (('whole', 'Whole KV'), ('icl', 'ICL KV'), ('qc', 'QC-TTT'))
DARK, GOLD, RUST, GRID = '#2B2118', '#A8763E', '#6F1A07', '#E8E1DA'


def read_data(plan, output):
    with SNAPSHOT.open(newline='') as stream:
        snapshot = list(csv.DictReader(stream))
    annotated = [r for r in snapshot if r['working_document'] == 'complete']
    if len(annotated) != 838 or any(float(r['fact_cover_lower']) != float(r['fact_cover_upper']) for r in annotated):
        raise ValueError('Unexpected or ambiguous OLMo coverage cohort')
    coverage = {}
    for length in LENGTHS:
        group = [r for r in annotated if int(r['context_tokens']) == length]
        if not group:
            raise ValueError('Missing context length in evidence annotations')
        selected = 100*sum(float(r['fact_cover_lower']) == 1 for r in group)/len(group)
        coverage[length] = dict(n=len(group), whole=100., icl=selected, qc=selected)
    planned = {c['key']: c for c in plan['cases']}
    records = {}
    for path in sorted((output/'cases').glob('*.json')):
        row = json.loads(path.read_text())
        key = row['key']
        if key not in planned or key in records:
            raise ValueError('Unexpected or duplicated measurement')
        if any(not 0 <= row['attention'][name]['attention_mass'] <= 1 for name, _ in METHODS):
            raise ValueError('Invalid recorded target attention mass')
        if row['context_tokens'] != planned[key]['context_tokens']:
            raise ValueError('Recorded context differs from planned context')
        records[key] = row
    return coverage, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    coverage, records = read_data(plan, args.output)
    groups = defaultdict(list)
    for record in records.values():
        groups[record['context_tokens']].append(record)
    summary = []
    for name, _ in METHODS:
        for length in LENGTHS:
            cases = groups[length]
            scores = [100*r['attention'][name]['attention_mass'] for r in cases]
            summary.append(dict(method=name, context_tokens=length, attention_cases=len(scores),
                mean_target_attention_pct=sum(scores)/len(scores) if scores else None,
                complete_evidence_recall_pct=coverage[length][name],
                complete_evidence_recall_cases=coverage[length]['n']))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'figure_data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(summary)
    complete = len(records) == plan['case_count']
    save_status = dict(planned=plan['case_count'], completed=len(records), complete=complete,
        attention_n_by_context={str(n): len(groups[n]) for n in LENGTHS},
        coverage_n_by_context={str(n): coverage[n]['n'] for n in LENGTHS},
        qc_and_icl_share_selected_text=True)
    (args.output/'figure_status.json').write_text(json.dumps(save_status, indent=2)+'\n')
    if not records:
        print('No attention measurements: data table saved, no figure generated.')
        return
    max_attention = max(100*r['attention'][name]['attention_mass'] for r in records.values() for name, _ in METHODS)
    # One fixed left scale across all methods; the independent right axis is 0–100%.
    ytop = max(.01, math.ceil(max_attention*1.2*100)/100)
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':8,
                         'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'}):
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.65), sharey=True, facecolor='white')
        fig.subplots_adjust(left=.11, right=.89, bottom=.20, top=.77, wspace=.12)
        for ax, (name, label) in zip(axes, METHODS):
            ax.set_facecolor('white')
            ax.set_title(label, fontsize=10, fontweight='bold', color=DARK, pad=10)
            ax.set(xlim=(-.45, 3.45), ylim=(0, ytop))
            ax.grid(axis='y', color=GRID, linewidth=.65)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(DARK); ax.spines[side].set_linewidth(.7)
            ax.tick_params(colors=DARK, labelsize=7.8, length=3, width=.7)
            twin = ax.twinx()
            # Padding keeps the 100% markers visible instead of clipping them.
            twin.set_ylim(0, 108)
            twin.set_yticks((0, 25, 50, 75, 100))
            twin.spines['top'].set_visible(False)
            twin.spines['left'].set_visible(False)
            twin.spines['bottom'].set_visible(False)
            twin.spines['right'].set_color(DARK)
            twin.tick_params(axis='y', colors=DARK, labelsize=7.8, length=3, width=.7)
            if ax is not axes[-1]:
                twin.spines['right'].set_visible(False)
                twin.tick_params(axis='y', right=False, labelright=False)
            if ax is not axes[0]:
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', left=False, labelleft=False)
            for j, length in enumerate(LENGTHS):
                group = sorted(groups[length], key=lambda r:r['key'])
                if group:
                    values = [100*r['attention'][name]['attention_mass'] for r in group]
                    offsets = np.linspace(-.16, .02, len(values)) if len(values)>1 else np.array([-.07])
                    ax.scatter(j+offsets, values, s=13, color=GOLD, alpha=.43,
                               linewidths=0, zorder=3)
                    avg = sum(values)/len(values)
                    ax.scatter([j-.07], [avg], s=42, color=GOLD, marker='o',
                               edgecolor='white', linewidth=.6, zorder=4)
                recall = coverage[length][name]
                twin.scatter([j+.20], [recall], s=44, facecolor=RUST, marker='D',
                             edgecolor='white', linewidth=.6, zorder=4)
            ax.set_xticks(range(4), ('8K','16K','32K','64K'))
            ax.tick_params(axis='x', length=0, pad=5)
        fig.text(.025, .50, 'Target attention mass (%)', color=GOLD, rotation=90,
                 va='center', ha='center', fontsize=9)
        fig.text(.975, .50, 'Complete-evidence recall (%)', color=RUST, rotation=270,
                 va='center', ha='center', fontsize=9)
        fig.text(.50, .08, 'Context length', ha='center', color=DARK, fontsize=9)
        handles = [Line2D([],[], marker='o', color=GOLD, linestyle='', markersize=5,
                          label='Target attention mass (qTTT)'),
                   Line2D([],[], marker='D', color=RUST, linestyle='', markersize=5,
                          label='Complete-evidence recall')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.97), ncol=2,
                   frameon=False, fontsize=8, handletextpad=.35, columnspacing=1.8)
        if not complete:
            fig.text(.5, .015, f"Preliminary: {len(records)}/{plan['case_count']} attention cases",
                     ha='center', color=DARK, fontsize=7)
        stem = 'motivation_1_official_metrics' + ('' if complete else '_partial')
        for ext in ('pdf','svg','png'):
            fig.savefig(args.output/f'{stem}.{ext}', dpi=600, facecolor='white')
        svg = args.output/f'{stem}.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        png = args.output/f'{stem}.png'
        with Image.open(png) as im:
            im.convert('RGB').save(png, dpi=(600,600))
        plt.close(fig)
    print(json.dumps(save_status, indent=2))


if __name__ == '__main__':
    main()
