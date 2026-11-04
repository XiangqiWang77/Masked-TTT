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
        output[name]=dict(answer_accuracy_pct=100*sum(int(r[score]) for r in rows)/len(rows),
            target_attention_pct=100*statistics.mean(float(r[name+'_target_attention_mass']) for r in attention),
            multihop_accuracy_pct=100*sum(int(r[score]) for r in multihop)/len(multihop))
        if output[name]['target_attention_pct']>1:
            raise ValueError('Display scale needs to be adjusted for the measured attention')
    return output


def main():
    metrics=load()
    vertices=np.array([[0.,1.],[-math.sqrt(3)/2,-.5],[math.sqrt(3)/2,-.5]])
    fig=plt.figure(figsize=(4.8,4.8),facecolor='white')
    ax=fig.add_axes((.08,.23,.84,.65));ax.set_aspect('equal');ax.axis('off')
    ax.set(xlim=(-1.30,1.30),ylim=(-.80,1.28))
    for radius in (.25,.5,.75,1.):
        points=np.vstack((vertices*radius,vertices[0]*radius))
        ax.plot(points[:,0],points[:,1],color='#E6E2DC',linewidth=.65,zorder=0)
    for vertex in vertices:
        ax.plot([0,vertex[0]],[0,vertex[1]],color='#DDD6CE',linewidth=.65,zorder=0)
    qc=np.array([metrics['qc']['answer_accuracy_pct']/100,
                 metrics['qc']['target_attention_pct']/1,
                 metrics['qc']['multihop_accuracy_pct']/100])
    qp=vertices*qc[:,None]
    ax.fill(qp[:,0],qp[:,1],color='#6F1A07',alpha=.09,zorder=1)
    for name,label,score,color,marker in STYLES:
        m=metrics[name]
        values=np.array([m['answer_accuracy_pct']/100,m['target_attention_pct']/1,
                         m['multihop_accuracy_pct']/100])
        points=vertices*values[:,None]
        loop=np.vstack((points,points[0]))
        ax.plot(loop[:,0],loop[:,1],color=color,linewidth=1.65 if name=='qc' else 1.25,
                linestyle=(0,(3,2)) if name=='icl' else '-',zorder=3 if name=='qc' else 2)
        # An outlined ICL square remains visible around the nearly overlapping QC diamond.
        ax.scatter(points[:,0],points[:,1],s=47 if name=='icl' else 32,marker=marker,
                   facecolor='white' if name=='icl' else color,edgecolor=color,
                   linewidth=1.,zorder=4 if name=='icl' else 5)
    ax.text(0,1.12,'Answer accuracy\n0–100%',ha='center',va='bottom',color='#2B2118',fontsize=8.1)
    ax.text(-.65,-.68,'Dilution ↓\nevidence attention 0–1%',ha='center',va='center',color='#2B2118',fontsize=7.1)
    ax.text(.65,-.68,'Fragmentation check\nQA2–3 accuracy 0–100%',ha='center',va='center',color='#2B2118',fontsize=7.1)

    # A compact numerical key avoids reading exact values from overlapping vertices.
    columns=(.15,.48,.68,.87)
    fig.text(columns[0],.215,'',fontsize=7)
    for x,label in zip(columns[1:],('Accuracy','Attention','QA2–3')):
        fig.text(x,.215,label,ha='center',color='#75695F',fontsize=7.2)
    for y,(name,label,_,color,marker) in zip((.171,.132,.093),STYLES):
        fig.text(columns[0],y,label,color=color,fontsize=8.0,fontweight='bold' if name=='qc' else 'normal')
        vals=metrics[name]
        for x,formatted in zip(columns[1:],(f"{vals['answer_accuracy_pct']:.1f}%",
                f"{vals['target_attention_pct']:.2f}%",f"{vals['multihop_accuracy_pct']:.1f}%")):
            fig.text(x,y,formatted,ha='center',color=color,fontsize=8.0,
                     fontweight='bold' if name=='qc' else 'normal')
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
        radial_display_maxima=dict(answer_accuracy_pct=100,target_attention_pct=1,
                                   multihop_accuracy_pct=100),
        axis_interpretation=dict(dilution='lower dilution is represented by greater target attention',
            fragmentation='QA2 and QA3 task accuracy is a functional diagnostic of multi-fact question answering, not a direct measurement of fragmentation'),
        radial_polygon_area_is_not_a_composite_score=True,
        selected_icl_and_qc_have_same_textual_evidence=True,
        teacher_and_qc_answer_formats_differ=True)
    (ROOT/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
