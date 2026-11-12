"""Recompute the two three-way figures from saved records and annotations.

No model calls, sampling, GPU execution, or statistical extrapolation.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

ARMS = ('agnostic', 'evidence_only', 'specific')
DATASETS = ('longhealth', 'mtob_ke', 'qasper')
NAMES = {'longhealth':'LongHealth', 'mtob_ke':'MTOB (KE)', 'qasper':'QASPER', 'all':'Overall'}
LABELS = {'agnostic':'Query-agnostic', 'evidence_only':'Evidence-only', 'specific':'Query-specific'}
COLORS = {'agnostic':'#3180A8', 'evidence_only':'#6EC1D8', 'specific':'#E36A1C'}
INK = '#0B2A3A'


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def collect(source):
    spec = read(source/'metric_specification.json')
    stop = set(spec['stopwords'])
    words = lambda text: set(re.findall(spec['word_pattern'], text.casefold()))-stop
    with gzip.open(source/'records.jsonl.gz', 'rt', encoding='utf8') as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 4500
    index, grouped = {}, defaultdict(list)
    for r in records:
        key = (r['dataset'], r['case'], r['arm'], r['question_index'])
        assert key not in index
        assert r['covered'] == int(bool(words(r['query']) & words(r['question'])))
        assert r['shared_terms'] == sorted(words(r['query']) & words(r['question']))
        index[key] = r
        grouped[key[:3]].append(r['covered'])
    assert len(grouped) == 300 and all(len(v)==15 for v in grouped.values())
    requests = {(ds, case) for ds,case,arm in grouped}
    assert len(requests) == 100
    assert all((ds,case,arm) in grouped for ds,case in requests for arm in ARMS)
    coverage = []
    for ds in (*DATASETS, 'all'):
        for arm in ARMS:
            values = [100*statistics.mean(v) for (dataset,case,a),v in grouped.items()
                      if a==arm and (ds=='all' or ds==dataset)]
            coverage.append(dict(dataset=ds, arm=arm, requests=len(values),
                questions=15*len(values), query_term_coverage_pct=statistics.mean(values),
                lexical_dilution_pct=100-statistics.mean(values)))
    with gzip.open(source/'reviewed_items.jsonl.gz', 'rt', encoding='utf8') as handle:
        reviews = [json.loads(line) for line in handle]
    assert len(reviews)==900
    reviewed_keys, counts, by_case = set(), defaultdict(Counter), Counter()
    for r in reviews:
        key=(r['dataset'],r['case'],r['arm'],r['question_index'])
        assert key not in reviewed_keys and key in index
        reviewed_keys.add(key)
        assert r['synthetic_question']==index[key]['question']
        assert r['real_query']==index[key]['query']
        assert r['label'] in ('D','P','O','U') and r['reason']
        by_case[key[:3]]+=1
        for ds in (r['dataset'],'all'):
            counts[ds,r['arm']][r['label']]+=1
    assert set(by_case)==set(grouped) and all(n==3 for n in by_case.values())
    for ds,case in requests:
        selected=[{key[3] for key in reviewed_keys if key[:3]==(ds,case,arm)} for arm in ARMS]
        assert selected[0]==selected[1]==selected[2]
    rates=[]
    for ds in (*DATASETS,'all'):
        for arm in ARMS:
            c=counts[ds,arm]; n=sum(c.values())
            rates.append(dict(dataset=ds,arm=arm,questions=n,requests=n//3,
                direct=c['D'],prerequisite=c['P'],off_target=c['O'],unclear=c['U'],
                off_target_pct=100*c['O']/n,unclear_pct=100*c['U']/n))
    return coverage,rates


def plot(coverage, rates, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    style={'font.family':'DejaVu Sans','font.size':8.3,'text.color':INK,
        'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,
        'axes.linewidth':.7,'hatch.linewidth':.45,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'}
    metadata={}
    def export(fig, name):
        for ext in ('png','pdf','svg'):
            fig.savefig(output/(name+'.'+ext),dpi=400,facecolor='white')
        plt.close(fig)
        with Image.open(output/(name+'.png')) as im:
            metadata[name]=dict(pixels=list(im.size),dpi=list(im.info['dpi']),
                square=im.width==im.height,
                opaque=im.convert('RGBA').getchannel('A').getextrema()==(255,255),
                width_inches=3.55,height_inches=3.55)
    cov={(r['dataset'],r['arm']):r for r in coverage}
    off={(r['dataset'],r['arm']):r for r in rates}
    with plt.rc_context(style):
        fig,ax=plt.subplots(figsize=(3.55,3.55))
        fig.subplots_adjust(left=.17,right=.985,bottom=.15,top=.80)
        for j,arm in enumerate(ARMS):
            xs=[i+(j-1)*.25 for i in range(3)]
            values=[cov[d,arm]['query_term_coverage_pct'] for d in DATASETS]
            ax.bar(xs,values,width=.215,color=COLORS[arm],edgecolor=INK,linewidth=.4,
                hatch=('', '..', '//')[j],label=LABELS[arm],zorder=3)
            for x,value in zip(xs,values):
                ax.text(x,value+1.8,f'{value:.1f}',ha='center',va='bottom',fontsize=7.0)
        ax.set(xlim=(-.55,2.55),ylim=(0,110),ylabel='Query-term coverage (%)')
        ax.set_xticks(range(3),[NAMES[d] for d in DATASETS]);ax.set_yticks([0,25,50,75,100])
        ax.tick_params(axis='x',length=0,pad=7,labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',color='#DFE7EB',linewidth=.5)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.56,.985),
            ncol=1,frameon=False,fontsize=8.3,labelspacing=.35,handlelength=1.6)
        export(fig,'synthetic_dilution_coverage')

        fig,ax=plt.subplots(figsize=(3.55,3.55))
        fig.subplots_adjust(left=.29,right=.985,bottom=.16,top=.75)
        # Matplotlib's s is proportional to marker area, never radius.
        scale=8.0
        for j,arm in enumerate(ARMS):
            for i,ds in enumerate(DATASETS):
                value=off[ds,arm]['off_target_pct']
                if value>0:
                    ax.scatter(i,j,s=scale*value,color=COLORS[arm],edgecolors=INK,linewidths=.65,zorder=3)
                ax.text(i,j+.39,f'{value:.1f}%',ha='center',va='center',fontsize=8.4,color=INK)
        ax.set(xlim=(-.55,2.55),ylim=(2.57,-.57))
        ax.set_xticks(range(3),[NAMES[d] for d in DATASETS])
        ax.set_yticks(range(3),['Query-\nagnostic','Evidence-\nonly','Query-\nspecific'])
        ax.tick_params(axis='both',length=0,pad=8,labelsize=8.2)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_axisbelow(True);ax.grid(color='#E6EDF0',linewidth=.5)
        # A dedicated size key reserves physical room for large markers.
        from matplotlib.lines import Line2D
        fig.text(.64,.975,'Off-target rate (%)',ha='center',va='top',fontsize=8.5)
        for x,value in zip((.43,.64,.85),(25,50,100)):
            fig.add_artist(Line2D([x],[.87],transform=fig.transFigure,linestyle='none',
                marker='o',markersize=(scale*value)**.5,markerfacecolor='white',
                markeredgecolor=INK,markeredgewidth=.6))
            fig.text(x,.785,f'{value}%',ha='center',va='center',fontsize=8)
        export(fig,'synthetic_off_target_bubbles')
    metadata.update(matplotlib_version=matplotlib.__version__,colors=COLORS,
        ink=INK,figure_count=2,bubble_count=9,bubble_area_points_squared='8 * off_target_percentage',
        confidence_intervals=None,coverage_axis_minimum=0,smoothing=None)
    return metadata


def main(source, output, verify_only):
    coverage,rates=collect(source)
    if verify_only:
        print('Verified 4,500 coverage records and 900 annotations on 100 matched requests.')
        return
    output.mkdir(parents=True,exist_ok=True)
    dump(output/'summary.json',dict(coverage=coverage,off_target=rates))
    for filename,rows in [('coverage_table.csv',coverage),('off_target_table.csv',rates)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata=plot(coverage,rates,output)
    dump(output/'figure_metadata.json',metadata)
    print(json.dumps(dict(output=str(output),**metadata)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();main(args.source,args.output or args.source,args.verify_only)
