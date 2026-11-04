#!/usr/bin/env python3
"""Two reproducible, matched motivation figures from a fixed scored snapshot."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parent
MODELS=('Qwen3-4B','GPT-OSS-20B')
SIZES=(8192,16384,32768,65536)
METHODS=('whole','selected','cartridge')
STYLE={
    'whole':dict(label='Full KV',color='#2B2118',marker='o',linestyle='-',fillstyle='full'),
    'selected':dict(label='Selected KV',color='#A8763E',marker='s',linestyle='--',fillstyle='none'),
    'cartridge':dict(label='Cartridge',color='#6F1A07',marker='^',linestyle='-',fillstyle='full'),
}


def load():
    with (ROOT/'data_snapshot.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        row['length']=int(row['length'])
        for key in ('whole','selected','cartridge'):row[key]=int(row[key])
        for key in ('fact_cover_lower','fact_cover_upper'):row[key]=float(row[key])
    if len({(row['model'],row['case']) for row in rows})!=len(rows):
        raise ValueError('Duplicate model/case pair')
    return rows


def scale_rows(rows):
    result={}
    for model in MODELS:
        sources=defaultdict(dict)
        for row in rows:
            if row['model']!=model:continue
            bucket=sources[row['source_question']]
            if row['length'] in bucket:raise ValueError('Duplicate source question at one length')
            bucket[row['length']]=row
        matched=[group for group in sources.values() if set(group)==set(SIZES)]
        if not matched:raise ValueError('No complete four-length source questions')
        result[model]={length:[group[length] for group in matched] for length in SIZES}
    return result


def coverage_rows(rows):
    result={}
    for model in MODELS:
        groups={'0%':[],'1–99%':[],'100%':[]}
        for row in rows:
            if row['model']!=model:continue
            low,high=row['fact_cover_lower'],row['fact_cover_upper']
            if low!=high:raise ValueError('Ambiguous coverage is not silently assigned a group')
            label='0%' if low==0 else '100%' if low==1 else '1–99%'
            groups[label].append(row)
        if any(not group for group in groups.values()):raise ValueError('A coverage group is empty')
        result[model]=groups
    return result


def rate(rows,method):
    return 100*sum(row[method] for row in rows)/len(rows)


def canvas(y_limits,y_ticks):
    fig,axes=plt.subplots(2,1,figsize=(3.55,3.55),sharex=True,facecolor='white')
    fig.subplots_adjust(left=.19,right=.97,bottom=.16,top=.87,hspace=.28)
    for ax,model in zip(axes,MODELS):
        ax.set_facecolor('white');ax.set_ylim(*y_limits);ax.set_yticks(y_ticks)
        ax.grid(axis='y',color='#DFDBD5',linewidth=.55,zorder=0)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        for name in ('bottom','left'):
            ax.spines[name].set_color('#403931');ax.spines[name].set_linewidth(.7)
        ax.tick_params(axis='both',labelsize=7.5,length=3,width=.65,colors='#2B2118')
        ax.text(.025,.955,model,transform=ax.transAxes,fontsize=8,fontweight='semibold',
                ha='left',va='top',color='#2B2118')
    handles=[]
    for method in METHODS:
        s=STYLE[method]
        handles.append(Line2D([],[],color=s['color'],linestyle=s['linestyle'],marker=s['marker'],
            markerfacecolor='white' if s['fillstyle']=='none' else s['color'],
            markeredgecolor=s['color'],linewidth=1.35,markersize=4.1,label=s['label']))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,.985),ncol=3,
        frameon=False,fontsize=7.2,handlelength=1.65,handletextpad=.45,columnspacing=.8)
    fig.text(.035,.51,'Answer accuracy (%)',rotation=90,ha='center',va='center',
             fontsize=8.1,color='#2B2118')
    return fig,axes


def plot_lines(ax,x,groups,offsets=None):
    for method in METHODS:
        s=STYLE[method];values=[rate(group,method) for group in groups]
        shifted=[position+(offsets or {}).get(method,0.) for position in x]
        ax.plot(shifted,values,color=s['color'],linestyle=s['linestyle'],marker=s['marker'],
            markerfacecolor='white' if s['fillstyle']=='none' else s['color'],
            markeredgecolor=s['color'],markeredgewidth=1.15,linewidth=1.4,markersize=4.4,
            zorder=4 if method=='cartridge' else 3)


def save(fig,stem):
    output=ROOT/'figures';output.mkdir(exist_ok=True)
    for extension in ('png','pdf','svg'):
        fig.savefig(output/(stem+'.'+extension),dpi=600,facecolor='white',transparent=False,
                    metadata={'Title':stem} if extension!='svg' else None)
    path=output/(stem+'.png')
    with Image.open(path) as picture:
        if picture.mode!='RGBA' or picture.getchannel('A').getextrema()!=(255,255):
            raise ValueError('Expected an opaque white-background rendering')
        picture.convert('RGB').save(path,dpi=(600,600))
    vector=output/(stem+'.svg')
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    rows=load();scaled=scale_rows(rows);covered=coverage_rows(rows);summary=[]

    fig,axes=canvas((20,80),(20,40,60,80))
    for ax,model in zip(axes,MODELS):
        groups=[scaled[model][length] for length in SIZES]
        plot_lines(ax,range(4),groups)
        ax.set_xlim(-.20,3.20)
        for length,group in zip(SIZES,groups):
            for method in METHODS:
                summary.append(dict(figure='context_scale',model=model,group=f'{length//1024}K',
                    n=len(group),method=method,correct=sum(row[method] for row in group),
                    accuracy_percent=rate(group,method)))
    axes[-1].set_xticks(range(4),('8K','16K','32K','64K'))
    axes[-1].set_xlabel('Context length (tokens)',fontsize=8.1,color='#2B2118',labelpad=5)
    save(fig,'motivation_1_context_scale')

    fig,axes=canvas((0,100),(0,25,50,75,100))
    labels=('0%','1–99%','100%')
    for ax,model in zip(axes,MODELS):
        groups=[covered[model][label] for label in labels]
        plot_lines(ax,range(3),groups,offsets={'whole':-.05,'selected':0.,'cartridge':.05})
        ax.set_xlim(-.23,2.23)
        for label,group in zip(labels,groups):
            for method in METHODS:
                summary.append(dict(figure='evidence_coverage',model=model,group=label,
                    n=len(group),method=method,correct=sum(row[method] for row in group),
                    accuracy_percent=rate(group,method)))
    axes[-1].set_xticks(range(3),labels)
    axes[-1].set_xlabel('Selected support fact coverage',fontsize=8.1,color='#2B2118',labelpad=5)
    save(fig,'motivation_2_evidence_coverage')

    with (ROOT/'plot_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summary)
    print('Wrote both figures and plot_summary.csv')


if __name__=='__main__':main()
