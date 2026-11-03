#!/usr/bin/env python3
"""Render the locked snapshot; every plotted value is available in the JSON."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE/"data_snapshot.json").read_text())
DEST=HERE/"figures"
DEST.mkdir(exist_ok=True)
NAVY="#003049"
GOLD="#A8763E"
RUST="#6F1A07"
INK="#2B2118"
GRID="#DDDAD4"
CREAM="#F7F3E3"
matplotlib.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
    "text.color":INK,"pdf.fonttype":42,"svg.fonttype":"none",
    "savefig.facecolor":"white","figure.facecolor":"white",
})


def save(fig,name):
    for suffix in ("png","pdf","svg"):
        fig.savefig(DEST/(name+"."+suffix),dpi=320,facecolor="white")
    plt.close(fig)


def figure_coverage():
    rows=DATA["coverage_audit_rows"]
    fig,ax=plt.subplots(figsize=(5.8,5.8),layout="constrained")
    for group,color,marker,linestyle,label in [
        ("complete",NAVY,"o","-","All supporting facts selected"),
        ("incomplete",RUST,"D","--","Some supporting facts omitted")]:
        group_rows=[r for r in rows if r["coverage"]==group]
        n=len(group_rows)
        scores=[100*sum(r[k] for r in group_rows)/n for k in ("full_zero","selected_zero")]
        ax.plot([0,1],scores,color=color,marker=marker,linewidth=2.3,
                markersize=8,linestyle=linestyle,label=f"{label}  (n={n})")
        for x,y in zip((0,1),scores):
            ax.annotate(f"{y:.1f}",(x,y),xytext=(0,10),textcoords="offset points",
                        ha="center",va="bottom",fontsize=11,color=color,fontweight="bold")
    ax.set_xlim(-.25,1.25);ax.set_ylim(0,100)
    ax.set_xticks([0,1],["Full-document\ncache","Selected-evidence\ncache"])
    ax.set_yticks([0,20,40,60,80,100])
    ax.set_ylabel("Answer accuracy (%)")
    ax.grid(axis="y",color=GRID,linewidth=.7,alpha=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left",frameon=False,fontsize=9.2,handlelength=2.6)
    save(fig,"01_evidence_coverage_tradeoff")


def figure_kl():
    fig,ax=plt.subplots(figsize=(5.8,5.8),layout="constrained")
    low,high=2e-5,1
    axis=np.geomspace(low,high,300)
    ax.fill_between(axis,low,axis,color=GOLD,alpha=.075,zorder=0)
    ax.plot(axis,axis,color="#776f67",linewidth=1.1,linestyle="--",zorder=1)
    for weight,color,marker in [(.25,NAVY,"o"),(.75,RUST,"D")]:
        rows=[r for r in DATA["kl_same_prefix"] if r["weight"]==weight]
        lower=sum(r["after"]<r["before"] for r in rows)
        ax.scatter([r["before"] for r in rows],[r["after"] for r in rows],
                   s=51,c=color,marker=marker,alpha=.72,edgecolors="white",
                   linewidths=.55,label=rf"$\lambda={weight:.2f}$   ({lower}/{len(rows)} decreased)",zorder=3)
    ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlim(low,high);ax.set_ylim(low,high);ax.set_aspect("equal",adjustable="box")
    ax.set_xlabel("Full-to-selected KL before update (nats/token)")
    ax.set_ylabel("Full-to-selected KL after one update (nats/token)")
    ax.grid(which="major",color=GRID,linewidth=.7,alpha=.75)
    ax.legend(loc="upper left",frameon=True,facecolor="white",edgecolor="none",
              framealpha=.94,fontsize=9)
    save(fig,"02_same_prefix_kl")


def figure_paired_answers():
    rows=DATA["rows"]
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.8),layout="constrained")
    cmap=LinearSegmentedColormap.from_list("evidence",["#F7F3E3","#DDC49D","#A8763E","#6F1A07"])
    matrices=[]
    for ax,before,after,title in zip(axes,["full_before","selected_before"],
                                     ["full_after","selected_after"],
                                     ["Full-document cache","Selected-evidence cache"]):
        mat=np.array([[sum(r[before]==i and r[after]==j for r in rows)
                       for j in (0,1)] for i in (0,1)])
        matrices.append(mat)
        ax.imshow(mat,cmap=cmap,vmin=0,vmax=len(rows),aspect="equal")
        for i in (0,1):
            for j in (0,1):
                value=int(mat[i,j]);color="white" if value>=10 else INK
                ax.text(j,i,str(value),ha="center",va="center",fontsize=21,fontweight="bold",color=color)
        ax.set_title(title,fontsize=11.2,pad=10,fontweight="bold")
        ax.set_xticks([0,1],["Wrong","Correct"]);ax.set_yticks([0,1],["Wrong","Correct"])
        ax.set_xlabel("After training")
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_xticks([-.5,.5,1.5],minor=True);ax.set_yticks([-.5,.5,1.5],minor=True)
        ax.grid(which="minor",color="white",linewidth=2.5);ax.tick_params(which="minor",length=0)
    axes[0].set_ylabel("Before training")
    axes[1].set_ylabel("Before training")
    save(fig,"03_paired_answer_outcomes")


def figure_storage():
    rows=DATA["rows"]
    fig,ax=plt.subplots(figsize=(5.8,5.8),layout="constrained")
    rng=random.Random(20261103)
    buckets=[8192,16384,32768,65536]
    ratios=[100*r["standalone_kv_and_adapter_bytes"]/r["full_document_kv_bytes"] for r in rows]
    for index,bucket in enumerate(buckets):
        group=[r for r in rows if r["length"]==bucket]
        xs=[index+rng.uniform(-.11,.11) for _ in group]
        ys=[100*r["standalone_kv_and_adapter_bytes"]/r["full_document_kv_bytes"] for r in group]
        ax.scatter(xs,ys,s=63,c=NAVY,alpha=.72,edgecolors="white",linewidths=.7,zorder=3)
    median=statistics.median(ratios)
    ax.axhline(median,color=RUST,linewidth=1.6,linestyle="--",label=f"Median: {median:.1f}%")
    ax.set_ylim(0,100);ax.set_xlim(-.55,3.55)
    ax.set_xticks(range(4),["8K","16K","32K","64K"])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Cartridge storage / full-document KV (%)")
    ax.grid(axis="y",color=GRID,linewidth=.7,alpha=.8)
    ax.legend(loc="upper right",frameon=False)
    save(fig,"04_standalone_storage")


if __name__=="__main__":
    figure_coverage();figure_kl();figure_paired_answers();figure_storage()
    print("Saved four figures in PNG, PDF, and SVG:",DEST)
