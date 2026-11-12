"""Recompute the published record diagnostics and redraw the figure; no model needed."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

DATASETS = ('longhealth', 'mtob_ke', 'qasper')
NAMES = {'longhealth':'LongHealth', 'mtob_ke':'MTOB (KE)', 'qasper':'QASPER'}
ARMS = ('vanilla', 'specific')
METRICS = ('query_term_focus', 'disjoint_source_context')
BLUE, ORANGE, INK = '#3180A8', '#E36A1C', '#0B2A3A'

def merged(spans):
    out = []
    for start, end in sorted(spans):
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end):
            raise ValueError('Invalid source span')
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out

def summarize(case_rows, records):
    out = {}
    for dataset in DATASETS:
        subset = [r for r in case_rows if r['dataset'] == dataset]
        vr = [r for r in subset if r['arm'] == 'vanilla']
        out[dataset] = dict(requests=len(vr),
            documents=len({r['document_key'] for r in vr}),
            matched_exercise_exposures_per_arm=sum(r['questions'] for r in vr),
            query_fallback_requests=sum(r['paired_request_used_query_fallback'] for r in vr),
            arms={})
        for arm in ARMS:
            rows = [r for r in subset if r['arm'] == arm]
            generated = [r for r in rows if not r['paired_request_used_query_fallback']]
            actual = [r for r in records if r['dataset'] == dataset and r['arm'] == arm]
            out[dataset]['arms'][arm] = dict(
                **{metric: 100 * statistics.mean(r[metric] for r in rows)
                   for metric in (*METRICS, 'query_term_recall', 'exact_query_copy')},
                unique_source_records=len({r['raw_record_id'] for r in actual}),
                excluding_query_fallback=dict(requests=len(generated), **{
                    metric: 100 * statistics.mean(r[metric] for r in generated)
                    if generated else None for metric in METRICS}))
    return out

def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    datasets = ('longhealth', 'mtob_ke', 'qasper')
    names = ('LongHealth', 'MTOB (KE)', 'QASPER')
    ink = '#0B2A3A'
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':8.5,
        'text.color':ink, 'axes.labelcolor':ink, 'axes.edgecolor':ink,
        'xtick.color':ink, 'ytick.color':ink, 'axes.linewidth':.7,
        'hatch.linewidth':.4, 'pdf.fonttype':42, 'ps.fonttype':42,
        'svg.fonttype':'none', 'figure.facecolor':'white', 'savefig.facecolor':'white'}):
        fig, ax = plt.subplots(figsize=(3.35,3.35))
        fig.subplots_adjust(left=.18, right=.985, bottom=.16, top=.79)
        for arm, offset, color, hatch, label in (
            ('vanilla', -.17, '#3180A8', '', 'Vanilla Cartridges'),
            ('specific', .17, '#E36A1C', '//', 'Contextual Cartridges')):
            values = [summary[d]['arms'][arm]['query_term_focus'] for d in datasets]
            xs = [i+offset for i in range(3)]
            ax.bar(xs, values, width=.29, color=color, edgecolor=ink,
                   linewidth=.4, hatch=hatch, label=label, zorder=3)
            for x, y in zip(xs,values):
                ax.text(x, y+2, f'{y:.1f}', ha='center', va='bottom', fontsize=8)
        ax.set(xlim=(-.55,2.55), ylim=(0,110), ylabel='Query-term coverage (%)')
        ax.set_xticks(range(3), names)
        ax.set_yticks([0,25,50,75,100])
        ax.tick_params(axis='x', length=0, pad=7, labelsize=8)
        ax.tick_params(axis='y', length=3)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis='y', color='#DFE7EB', linewidth=.5)
        fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=1,
            loc='upper center', bbox_to_anchor=(.56,.985), fontsize=8.5,
            handlelength=1.6, handleheight=1.05, labelspacing=.5)
        stem = output/'synthetic_dilution_coverage'
        for extension in ('png','pdf','svg'):
            fig.savefig(stem.with_suffix('.'+extension), dpi=400, facecolor='white')
        plt.close(fig)
    with Image.open(stem.with_suffix('.png')) as im:
        raster = dict(pixels=list(im.size), dpi=list(im.info['dpi']), mode=im.mode,
                      square=im.width==im.height, opaque=im.convert('RGBA').getchannel('A').getextrema()==(255,255))
    return dict(width_inches=3.35, height_inches=3.35, png=raster,
        matplotlib_version=matplotlib.__version__, metric='query_term_focus',
        uncertainty='none', normalization='equal_request_macro_mean_percent',
        colors={'vanilla':'#3180A8','specific':'#E36A1C','ink':ink},
        zero_baseline=True, same_underlying_values=True,
        venue_compliance='general manuscript figure; no venue compliance claim')

def verify_and_collect(root):
    specification=json.loads((root/'metric_specification.json').read_text())
    token=re.compile(specification['word_pattern'],re.UNICODE)
    stop=set(specification['stopwords'])
    terms=lambda text:set(token.findall(text.casefold()))-stop
    normalize=lambda text:' '.join(token.findall(text.casefold()))
    with gzip.open(root/'records.jsonl.gz','rt',encoding='utf-8') as stream:
        records=[json.loads(line) for line in stream]
    groups=defaultdict(list)
    for row in records:
        question=row['query'] if row['is_query_fallback'] else row['question']
        assert question==row['metric_question']
        q=terms(row['query']); assert q
        hits=sorted(q&terms(question))
        assert hits==row['matched_query_terms'] and sorted(q)==row['query_terms']
        spans=merged(row['source_regions'])
        assert spans==row['source_regions']
        actual={'query_term_focus':bool(hits), 'query_term_recall':len(hits)/len(q),
                'disjoint_source_context':len(spans)>=2,
                'exact_query_copy':normalize(question)==normalize(row['query'])}
        for metric,value in actual.items():
            assert abs(float(row[metric])-float(value))<1e-12,(row['case'],metric)
        row.update(actual)
        groups[(row['dataset'],row['case'],row['arm'])].append(row)
    case_rows=[]
    for (dataset,case,arm),items in groups.items():
        assert [x['question_index'] for x in items]==list(range(len(items)))
        assert len(items)==len(groups[(dataset,case,'specific' if arm=='vanilla' else 'vanilla')])
        assert len({x['document_sha256'] for x in items})==1
        case_rows.append(dict(dataset=dataset,case=case,arm=arm,
            document_key=items[0]['document_key'],questions=len(items),
            paired_request_used_query_fallback=items[0]['paired_request_used_query_fallback'],
            **{metric:statistics.mean(x[metric] for x in items) for metric in
               (*METRICS,'query_term_recall','exact_query_copy')}))
    assert len(case_rows)==214 and len(records)==1284
    result=summarize(case_rows,records)
    expected=json.loads((root/'summary.json').read_text())
    def equal(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(int,float)):
            assert abs(a-b)<1e-9,(a,b)
        else:assert a==b
    equal(result,expected)
    return result,case_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('reproduced'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text())
    for name,fingerprint in manifest['artifact_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==fingerprint,name
    summary,case_rows=verify_and_collect(root)
    print('Verified 107 paired requests and 642 exercise exposures per method; all summary values reproduced.')
    if args.verify_only:return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    csv_write(args.output/'per_request.csv',case_rows)
    plot(summary,args.output)
    print('Figures written to',args.output.resolve())


if __name__=='__main__':main()
