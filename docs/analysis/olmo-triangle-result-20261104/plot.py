#!/usr/bin/env python3
"""Render three measured OLMo metrics on a transparent triangular radar."""
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
DOCS=ROOT.parent
SNAPSHOT=DOCS/'olmo-motivations-20261104/data_snapshot.csv'
ATTENTION=DOCS/'official-motivation-figure-20261104/data/target_attention_points.csv'
DEST=ROOT/'figures'
STYLES=(('whole','Whole KV','whole','#2B2118','o'),
        ('icl','ICL KV','teacher','#A8763E','s'),
        ('qc','QC-TTT','cartridge','#6F1A07','D'))


def load():
    with SNAPSHOT.open(newline='') as stream:rows=list(csv.DictReader(stream))
    with ATTENTION.open(newline='') as stream:attention=list(csv.DictReader(stream))
    if len(rows)!=1000 or len(attention)!=48:
        raise ValueError('Expected the frozen OLMo datasets')
    multihop=[r for r in rows if r['task'] in ('qa2','qa3')]
    if len(multihop)!=200:
        raise ValueError('Expected all 200 QA2 and QA3 requests')
    output={}
    for name,label,score,color,marker in STYLES:
        attention_mass=statistics.mean(float(r[name+'_target_attention_mass']) for r in attention)
        multihop_correct=sum(int(r[score]) for r in multihop)
        output[name]=dict(answer_accuracy_pct=100*sum(int(r[score]) for r in rows)/len(rows),
            target_attention_mass=attention_mass,target_attention_pct=100*attention_mass,
            multihop_correct=multihop_correct,
            multihop_accuracy_pct=100*multihop_correct/len(multihop))
        if output[name]['target_attention_pct']>1:
            raise ValueError('Display scale needs to be adjusted for the measured attention')
    return output


def main():
    metrics=load()
    keys=('answer_accuracy_pct','target_attention_pct','multihop_accuracy_pct')
    bounds={key:(min(metrics[name][key] for name,*_ in STYLES),
                 max(metrics[name][key] for name,*_ in STYLES)) for key in keys}
    # Common min-max transform per axis; the margins leave space for the data labels.
    def position(name):
        return np.array([.14+.72*(metrics[name][key]-bounds[key][0])/
                         (bounds[key][1]-bounds[key][0]) for key in keys])
    vertices=np.array([[0.,1.],[-math.sqrt(3)/2,-.5],[math.sqrt(3)/2,-.5]])
    fig=plt.figure(figsize=(4.8,4.8),facecolor='white')
    ax=fig.add_axes((.08,.13,.84,.76));ax.set_aspect('equal');ax.axis('off')
    ax.set(xlim=(-1.13,1.13),ylim=(-.76,1.18))
    for radius in (.14,.32,.50,.68,.86,1.):
        points=np.vstack((vertices*radius,vertices[0]*radius))
        ax.plot(points[:,0],points[:,1],color='#E6E2DC',linewidth=.65,zorder=0)
    for vertex in vertices:
        ax.plot([0,vertex[0]],[0,vertex[1]],color='#DDD6CE',linewidth=.65,zorder=0)
    qp=vertices*position('qc')[:,None]
    ax.fill(qp[:,0],qp[:,1],color='#6F1A07',alpha=.09,zorder=1)
    for name,label,score,color,marker in STYLES:
        points=vertices*position(name)[:,None]
        loop=np.vstack((points,points[0]))
        ax.plot(loop[:,0],loop[:,1],color=color,linewidth=1.65 if name=='qc' else 1.25,
                linestyle=(0,(3,2)) if name=='icl' else '-',zorder=3 if name=='qc' else 2)
        # An outlined ICL square remains visible around the nearly overlapping QC diamond.
        ax.scatter(points[:,0],points[:,1],s=47 if name=='icl' else 32,marker=marker,
                   facecolor='white' if name=='icl' else color,edgecolor=color,
                   linewidth=1.,zorder=4 if name=='icl' else 5)
    ax.text(0,1.045,'Answer accuracy ↑ (%)',ha='center',va='bottom',color='#2B2118',fontsize=8)
    ax.text(-.65,-.605,'Evidence attention ↑\n(dilution check)',ha='center',va='center',color='#2B2118',fontsize=7.4)
    ax.text(.65,-.605,'QA2–3 correct / 200 ↑\n(fragmentation check)',ha='center',va='center',color='#2B2118',fontsize=7.4)
    legend=[Line2D([0],[0],color=color,marker=marker,markersize=4.4,linewidth=1.25,
                   markerfacecolor='white' if name=='icl' else color,
                   linestyle=(0,(3,2)) if name=='icl' else '-',label=label)
            for name,label,_,color,marker in STYLES]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,
               frameon=False,fontsize=7.4,handlelength=1.5,columnspacing=1.5)

    # Direct labels show native units, never the normalized coordinates.
    label_locations={
        ('whole','answer_accuracy_pct'):(-.025,.17,'right'),
        ('icl','answer_accuracy_pct'):(-.03,.59,'right'),
        ('qc','answer_accuracy_pct'):(.025,.86,'left'),
        ('whole','target_attention_pct'):(-.15,-.005,'right'),
        ('icl','target_attention_pct'):(-.62,-.375,'center'),
        ('qc','target_attention_pct'):(-.62,-.455,'center'),
        ('whole','multihop_accuracy_pct'):(.64,-.47,'center'),
        ('icl','multihop_accuracy_pct'):(.16,-.09,'left'),
        ('qc','multihop_accuracy_pct'):(.54,-.22,'left'),
    }
    colors={name:color for name,_,_,color,_ in STYLES}
    for (name,key),(x,y,align) in label_locations.items():
        if key=='target_attention_pct':
            label=f"{metrics[name]['target_attention_mass']:.5f}"
        elif key=='multihop_accuracy_pct':
            label=f"{metrics[name]['multihop_correct']}/200"
        else:
            label=f"{metrics[name][key]:.1f}%"
        ax.text(x,y,label,ha=align,va='center',
                color=colors[name],fontsize=6.4,fontweight='bold' if name=='qc' else 'normal',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.6),zorder=7)
    fig.text(.5,.045,'Per-axis min–max · labels use original units',ha='center',
             color='#75695F',fontsize=6.8)
    DEST.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}):
        for ext in ('pdf','svg','png'):
            fig.savefig(DEST/f'olmo_three_axis.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    svg=DEST/'olmo_three_axis.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    png=DEST/'olmo_three_axis.png'
    with Image.open(png) as im:
        if im.mode=='RGBA' and im.getchannel('A').getextrema()!=(255,255):
            raise ValueError('Expected opaque white background')
        im.convert('RGB').save(png,dpi=(600,600))
    meta=dict(model='OLMo-3-7B',metric='answer_focused_lexical_v2',
        answer_accuracy_requests=1000,multihop_requests=200,target_attention_requests=48,
        target_attention_source_questions=12,metrics=metrics,
        normalization=dict(method='per-axis min-max across all three method means',
            radial_min=0.14,radial_max=0.86,
            raw_metric_bounds={key:list(bounds[key]) for key in keys},
            labels='raw accuracy percent, attention probability mass, and multi-hop correct count / 200'),
        axis_interpretation=dict(dilution='lower dilution is represented by greater target attention',
            fragmentation='QA2 and QA3 task accuracy is a functional diagnostic of multi-fact question answering, not a direct measurement of fragmentation'),
        radial_polygon_area_is_not_a_composite_score=True,
        selected_icl_and_qc_have_same_textual_evidence=True,
        teacher_and_qc_answer_formats_differ=True)
    (ROOT/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
