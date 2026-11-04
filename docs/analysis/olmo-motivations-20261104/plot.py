#!/usr/bin/env python3
"""Plot OLMo's recorded 1,000-case result and post-hoc evidence coverage."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parent
SIZES=(8192,16384,32768,65536)
METHODS=('whole','teacher','cartridge')
STYLE={
    'whole':dict(label='Full KV',color='#2B2118',marker='o',linestyle='-',open=False),
    'teacher':dict(label='ICL Teacher',color='#A8763E',marker='s',linestyle='--',open=True),
    'cartridge':dict(label='Cartridge',color='#6F1A07',marker='^',linestyle='-',open=False),
}


def load():
    with (ROOT/'data_snapshot.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        row['context_tokens']=int(row['context_tokens'])
        for key in METHODS:row[key]=int(row[key])
        for key in ('fact_cover_lower','fact_cover_upper'):
            row[key]=float(row[key]) if row[key] else None
    if len(rows)!=1000 or len({row['case'] for row in rows})!=1000:
        raise ValueError('The original OLMo snapshot must have 1,000 unique cases')
    return rows


def length_groups(rows):
    groups={length:[row for row in rows if row['context_tokens']==length] for length in SIZES}
    if any(len(group)!=250 for group in groups.values()):raise ValueError('Expected 250 per length')
    return groups


def coverage_groups(rows):
    groups={'0%':[],'1–99%':[],'100%':[]};excluded=0
    for row in rows:
        if row['working_document']!='complete':excluded+=1;continue
        low,high=row['fact_cover_lower'],row['fact_cover_upper']
        if low is None or low!=high or not 0<=low<=1:raise ValueError('Unknown fact coverage')
        key='0%' if low==0 else '100%' if low==1 else '1–99%'
        groups[key].append(row)
    if excluded!=162 or sum(map(len,groups.values()))!=838:
        raise ValueError('Unexpected document-context coverage')
    return groups


def rate(rows,method):
    return 100*sum(row[method] for row in rows)/len(rows)


def canvas(y_limits,y_ticks):
    fig,ax=plt.subplots(figsize=(3.55,3.55),facecolor='white')
    fig.subplots_adjust(left=.20,right=.97,bottom=.18,top=.86)
    ax.set_facecolor('white');ax.set_ylim(*y_limits);ax.set_yticks(y_ticks)
    ax.grid(axis='y',color='#DFDBD5',linewidth=.6,zorder=0)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    for name in ('bottom','left'):
        ax.spines[name].set_color('#403931');ax.spines[name].set_linewidth(.75)
    ax.tick_params(axis='both',labelsize=8,length=3,width=.7,colors='#2B2118')
    ax.set_ylabel('Answer accuracy (%)',fontsize=8.4,color='#2B2118',labelpad=5)
    handles=[]
    for method in METHODS:
        s=STYLE[method]
        handles.append(Line2D([],[],color=s['color'],linestyle=s['linestyle'],marker=s['marker'],
            markerfacecolor='white' if s['open'] else s['color'],markeredgecolor=s['color'],
            linewidth=1.5,markersize=4.5,label=s['label']))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,.985),ncol=3,
        frameon=False,fontsize=7.3,handlelength=1.55,handletextpad=.4,columnspacing=.8)
    return fig,ax


def draw(ax,groups,offsets=None):
    for method in METHODS:
        style=STYLE[method];x=[i+(offsets or {}).get(method,0) for i in range(len(groups))]
        ax.plot(x,[rate(group,method) for group in groups],color=style['color'],
            linestyle=style['linestyle'],marker=style['marker'],
            markerfacecolor='white' if style['open'] else style['color'],
            markeredgecolor=style['color'],markeredgewidth=1.2,linewidth=1.6,markersize=5.2,
            zorder=4 if method=='cartridge' else 3)


def save(fig,stem):
    output=ROOT/'figures';output.mkdir(exist_ok=True)
    for suffix in ('png','pdf','svg'):
        fig.savefig(output/(stem+'.'+suffix),dpi=600,facecolor='white',transparent=False,
                    metadata={'Title':stem} if suffix!='svg' else None)
    raster=output/(stem+'.png')
    with Image.open(raster) as picture:
        if picture.mode!='RGBA' or picture.getchannel('A').getextrema()!=(255,255):
            raise ValueError('Expected an opaque white-background rendering')
        picture.convert('RGB').save(raster,dpi=(600,600))
    vector=output/(stem+'.svg')
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    rows=load();length=length_groups(rows);coverage=coverage_groups(rows);summary=[]
    fig,ax=canvas((20,60),(20,30,40,50,60));groups=[length[size] for size in SIZES]
    draw(ax,groups);ax.set_xlim(-.20,3.20);ax.set_xticks(range(4),('8K','16K','32K','64K'))
    ax.set_xlabel('Context length (tokens)',fontsize=8.4,color='#2B2118',labelpad=6)
    for size,group in zip(SIZES,groups):
        for method in METHODS:
            summary.append(dict(figure='context_scale',group=f'{size//1024}K',n=len(group),
                method=method,correct=sum(row[method] for row in group),accuracy_percent=rate(group,method)))
    save(fig,'motivation_1_olmo_context_scale')

    labels=('0%','1–99%','100%');groups=[coverage[label] for label in labels]
    fig,ax=canvas((15,65),(20,30,40,50,60));draw(ax,groups,{'whole':-.045,'teacher':0.,'cartridge':.045})
    ax.set_xlim(-.23,2.23);ax.set_xticks(range(3),labels)
    ax.set_xlabel('Selected support fact coverage',fontsize=8.4,color='#2B2118',labelpad=6)
    for label,group in zip(labels,groups):
        for method in METHODS:
            summary.append(dict(figure='evidence_coverage',group=label,n=len(group),
                method=method,correct=sum(row[method] for row in group),accuracy_percent=rate(group,method)))
    save(fig,'motivation_2_olmo_evidence_coverage')

    with (ROOT/'plot_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summary)
    print('Wrote both OLMo figures and plot_summary.csv')


if __name__=='__main__':main()
