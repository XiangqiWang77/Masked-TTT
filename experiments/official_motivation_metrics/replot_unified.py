#!/usr/bin/env python3
"""Show evidence access, evidence coverage and observed QA outcomes together."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
SNAPSHOT=ROOT/'docs/analysis/olmo-motivations-20261104/data_snapshot.csv'
ATTENTION=ROOT/'docs/analysis/official-motivation-figure-20261104/data/target_attention_points.csv'
DEST=ROOT/'docs/analysis/qc-unified-motivation-20261104'
LENGTHS=(8192,16384,32768,65536)
METHODS=(('whole','Whole KV','whole','#2B2118'),
         ('icl','ICL KV','teacher','#A8763E'),
         ('qc','QC-TTT','cartridge','#6F1A07'))
DARK,GOLD,RUST,GRID,TRACK='#2B2118','#A8763E','#6F1A07','#EAE5DF','#ECF0F1'


def source_data():
    with SNAPSHOT.open(newline='') as f: original=list(csv.DictReader(f))
    with ATTENTION.open(newline='') as f: attention=list(csv.DictReader(f))
    if len(original)!=1000 or len(attention)!=48:
        raise ValueError('Expected the complete previously frozen datasets')
    selected=[r for r in original if r['working_document']=='complete']
    if len(selected)!=838 or any(float(r['fact_cover_lower'])!=float(r['fact_cover_upper']) for r in selected):
        raise ValueError('Unexpected support-annotation cohort')
    if Counter(int(r['context_tokens']) for r in attention)!=Counter({x:12 for x in LENGTHS}):
        raise ValueError('The attention sample must contain 12 cases per length')
    return selected,attention


def export(fig,stem):
    (DEST/'figures').mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(DEST/'figures'/f'{stem}.{ext}',dpi=600,facecolor='white')
    svg=DEST/'figures'/f'{stem}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    png=DEST/'figures'/f'{stem}.png'
    with Image.open(png) as im:
        if im.mode=='RGBA' and im.getchannel('A').getextrema()!=(255,255):
            raise ValueError('Expected an opaque background')
        im.convert('RGB').save(png,dpi=(600,600))
    plt.close(fig)


def draw_donut(fig,center_x,y,percent,color):
    # Identical radii and a 0–100% angular scale for all groups.
    ax=fig.add_axes((center_x-.041,y,.082,.162),aspect='equal')
    ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15));ax.axis('off')
    ax.add_patch(Wedge((0,0),1,0,360,width=.16,facecolor=TRACK,edgecolor='none'))
    ax.add_patch(Wedge((0,0),1,90-360*percent/100,90,width=.16,facecolor=color,edgecolor='none'))
    ax.text(0,0,f'{percent:.1f}%',color=color,ha='center',va='center',fontsize=8.1,fontweight='bold')


def main():
    rows,attention=source_data()
    by_group={}
    for name in ('all','complete','incomplete'):
        subset=[r for r in rows if (name=='all' or
                  (float(r['fact_cover_lower'])==1)==(name=='complete'))]
        by_group[name]={m:100*sum(int(r[score]) for r in subset)/len(subset)
                        for m,_,score,_ in METHODS}
        by_group[name]['n']=len(subset)
        by_group[name]['qc_new_correct_vs_icl']=sum(r['teacher']=='0' and r['cartridge']=='1' for r in subset)
        by_group[name]['qc_new_wrong_vs_icl']=sum(r['teacher']=='1' and r['cartridge']=='0' for r in subset)
    if [by_group[x]['n'] for x in ('all','complete','incomplete')]!=[838,524,314]:
        raise ValueError('Evidence-coverage groups changed')
    data=[]
    for name,label,score,color in METHODS:
        for length in LENGTHS:
            grp=[r for r in rows if int(r['context_tokens'])==length]
            att=[r for r in attention if int(r['context_tokens'])==length]
            masses=[100*float(r[name+'_target_attention_mass']) for r in att]
            recall=100. if name=='whole' else 100*sum(float(r['fact_cover_lower'])==1 for r in grp)/len(grp)
            data.append(dict(method=name,context_tokens=length,attention_n=len(att),
                target_attention_mass_pct=sum(masses)/len(masses),
                complete_evidence_recall_pct=recall,coverage_n=len(grp),
                answer_acc_pct=100*sum(int(r[score]) for r in grp)/len(grp)))
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'data').mkdir(exist_ok=True)
    with (DEST/'data/combined_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    (DEST/'data/outcomes_by_coverage.json').write_text(json.dumps(by_group,indent=2)+'\n')
    differences=[100*(float(r['qc_target_attention_mass'])-float(r['icl_target_attention_mass']))
                 for r in attention]
    epsilon_pp=.00001
    attention_changes=dict(cases=len(differences),threshold_pp=epsilon_pp,
        qc_greater=sum(d>epsilon_pp for d in differences),
        qc_smaller=sum(d<-epsilon_pp for d in differences),
        within_threshold=sum(abs(d)<=epsilon_pp for d in differences),
        median_absolute_change_pp=statistics.median(map(abs,differences)),
        note='Comparison of measured target attention; nonzero numerical changes are not proof of a causal QA benefit.')
    (DEST/'data/attention_changes.json').write_text(json.dumps(attention_changes,indent=2)+'\n')
    max_mass=max(100*float(r[m+'_target_attention_mass']) for r in attention for m,_,_,_ in METHODS)
    ymax=max(.01,math.ceil(max_mass*1.2*100)/100)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,3,figsize=(9.6,5.35),sharey=True,facecolor='white')
        fig.subplots_adjust(left=.11,right=.89,bottom=.16,top=.58,wspace=.12)
        fig.text(.5,.975,'ANSWER ACCURACY ON COMPLETE DOCUMENTS',ha='center',va='center',
                 color='#75695F',fontsize=8.0)
        for ax,(name,label,score,color) in zip(axes,METHODS):
            pos=ax.get_position();cx=(pos.x0+pos.x1)/2
            fig.text(cx,.93,label,color=color,fontsize=10.5,ha='center',va='center',fontweight='bold')
            draw_donut(fig,cx-.044,.745,by_group['complete'][name],color)
            draw_donut(fig,cx+.044,.745,by_group['incomplete'][name],color)
            fig.text(cx-.044,.735,'Selection:\nall facts',ha='center',va='top',color='#75695F',fontsize=6.8)
            fig.text(cx+.044,.735,'Selection:\nfacts missing',ha='center',va='top',color='#75695F',fontsize=6.8)
            acc=by_group['all'][name]
            extra=(f"   +{acc-by_group['all']['icl']:.1f} pp vs ICL" if name=='qc' else '')
            fig.text(cx,.675,f'Overall {acc:.1f}%{extra}',ha='center',va='center',
                     color=color,fontsize=8.0,fontweight='bold' if name=='qc' else 'normal')
            ax.set(xlim=(-.45,3.45),ylim=(0,ymax));ax.set_axisbelow(True)
            ax.grid(axis='y',color=GRID,linewidth=.55)
            for side in ('top','right'):ax.spines[side].set_visible(False)
            for side in ('left','bottom'):
                ax.spines[side].set_color(DARK);ax.spines[side].set_linewidth(.7)
            ax.tick_params(colors=DARK,labelsize=7.8,length=3,width=.7)
            twin=ax.twinx();twin.set_ylim(0,108);twin.set_yticks((0,25,50,75,100))
            for side in ('top','left','bottom'):twin.spines[side].set_visible(False)
            twin.spines['right'].set_color(DARK)
            twin.tick_params(axis='y',colors=DARK,labelsize=7.8,length=3,width=.7)
            if ax is not axes[-1]:
                twin.spines['right'].set_visible(False);twin.tick_params(axis='y',right=False,labelright=False)
            if ax is not axes[0]:
                ax.spines['left'].set_visible(False);ax.tick_params(axis='y',left=False,labelleft=False)
            for i,length in enumerate(LENGTHS):
                points=sorted((r for r in attention if int(r['context_tokens'])==length),key=lambda r:r['sample'])
                values=[100*float(r[name+'_target_attention_mass']) for r in points]
                spread=np.linspace(-.16,.02,len(values))
                ax.scatter(i+spread,values,s=13,color=GOLD,alpha=.40,linewidths=0,zorder=3)
                ax.scatter([i-.07],[sum(values)/len(values)],s=46,color=GOLD,
                           marker='o',edgecolor='white',linewidth=.6,zorder=4)
                data_row=next(r for r in data if r['method']==name and r['context_tokens']==length)
                twin.scatter([i+.20],[data_row['complete_evidence_recall_pct']],s=48,color=RUST,
                             marker='D',edgecolor='white',linewidth=.6,zorder=4)
            ax.set_xticks(range(4),('8K','16K','32K','64K'))
            ax.tick_params(axis='x',length=0,pad=6)
        fig.text(.025,.37,'Target attention mass (%)',color=GOLD,rotation=90,
                 va='center',ha='center',fontsize=9)
        fig.text(.975,.37,'Complete-evidence recall (%)',color=RUST,rotation=270,
                 va='center',ha='center',fontsize=9)
        fig.text(.5,.055,'Context length',ha='center',color=DARK,fontsize=9)
        handles=[Line2D([],[],marker='o',color=GOLD,linestyle='',markersize=5,
                        label='Target attention mass (qTTT)'),
                 Line2D([],[],marker='D',color=RUST,linestyle='',markersize=5,
                        label='Complete-evidence recall')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.635),ncol=2,
                   frameon=False,fontsize=7.8,handletextpad=.35,columnspacing=1.8)
        export(fig,'motivation_1_unified')
    print(json.dumps(by_group,indent=2))


if __name__=='__main__':main()
