"""Rebuild the ACL figure package from saved observations and explicit nulls.

No generation, training, scheduler access, smoothing, bars, or confidence bands.
The data snapshot preserves all 1,000 requests, including regressions.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

BLUE, ORANGE, INK = "#3180A8", "#D35B1D", "#0B2A3A"
PALETTE = ["#0B2A3A", "#1E4B6B", "#3180A8", "#6EC1D8", "#B7E3F1",
           "#F7C948", "#F59E2C", "#E36A1C", "#D35B1D", "#8C3A1A"]
LENGTHS = (8192, 16384, 32768, 65536)
ARMS = ("vanilla", "query_specific")
MODELS = ("Qwen3-4B", "OLMo-3-7B", "GPT-OSS-20B")
DATASETS = ("longhealth", "mtob_ke", "qasper")
METRICS = {"longhealth": ("LongHealth", "Accuracy (%)", "higher"),
           "mtob_ke": ("MTOB (KE)", "chrF", "higher"),
           "qasper": ("QASPER", "Log-perplexity", "lower")}
CACHES = (512, 1024, 2048, 4096)
STYLE = {"font.family": "DejaVu Sans", "font.size": 8,
         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
         "legend.fontsize": 7.5, "axes.linewidth": .65, "axes.edgecolor": INK,
         "axes.labelcolor": INK, "text.color": INK,
         "xtick.color": INK, "ytick.color": INK,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
         "figure.facecolor": "white", "savefig.facecolor": "white",
         "svg.hashsalt": "contextual-cartridges-acl-20261111"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n")


def rows_for(cases, metric, length=None, task=None, multi=False):
    return [c for c in cases if (length is None or c["length"] == length)
            and (task is None or c["task"] == task)
            and (not multi or c["multi_fact"])
            and all(c[a][metric] is not None for a in ARMS)]


def stats(cases, metric, **kwargs):
    rows = rows_for(cases, metric, **kwargs)
    if not rows:
        return {"n": 0, "vanilla": None, "query_specific": None, "delta": None}
    means = {a: float(np.mean([c[a][metric] for c in rows])) for a in ARMS}
    return {"n": len(rows), **means, "delta": means["query_specific"] - means["vanilla"]}


def axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=.6, pad=3)
    ax.grid(False)


def legend(fig):
    handles = [Line2D([], [], color=ORANGE, ls="--", marker="o", mfc="white",
                      ms=4, lw=1.4, label="Query-agnostic"),
               Line2D([], [], color=BLUE, marker="D", ms=3.8, lw=1.4,
                      label="Query-specific")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.54, 1.0), handlelength=2.4)


def curve_figure(cases):
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.50))
    fig.subplots_adjust(left=.105, right=.99, bottom=.22, top=.82, wspace=.47)
    for ax, metric, multi, ylabel in zip(axes, ("no_support", "joint"), (False, True),
            ("Contexts without support (%)", "Contexts with all facts (%)")):
        for a, color, marker, ls in zip(ARMS, (ORANGE, BLUE), ("o", "D"), ("--", "-")):
            y = [100*stats(cases, metric, length=l, multi=multi)[a] for l in LENGTHS]
            ax.plot(range(4), y, color=color, marker=marker, ls=ls, lw=1.6, ms=4.5,
                    mfc="white" if a == "vanilla" else color)
        ax.set(xticks=range(4), xticklabels=["8K", "16K", "32K", "64K"],
               xlabel="Document length", ylabel=ylabel, xlim=(-.15, 3.15))
        ax.set_ylim((0, 100) if metric == "no_support" else (0, 30))
        axis(ax)
    legend(fig)
    return fig


def heatmap_figure(cases):
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 3.55))
    fig.subplots_adjust(left=.10, right=.81, bottom=.15, top=.97, wspace=.43)
    cmap = LinearSegmentedColormap.from_list("blue_orange_gain",
            ["#8C3A1A", "#E36A1C", "#F7C948", "#FFFFFF", "#B7E3F1", "#3180A8", "#0B2A3A"])
    cmap.set_bad("#E8EAEC")
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    for ax, metric, multi in zip(axes, ("no_support", "joint"), (False, True)):
        z = np.full((10, 4), np.nan)
        for ti in range(10):
            for li, length in enumerate(LENGTHS):
                d = stats(cases, metric, task=f"qa{ti+1}", length=length, multi=multi)
                if d["n"]:
                    z[ti, li] = 100*d["delta"] * (-1 if metric == "no_support" else 1)
        im = ax.imshow(z, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        ax.set(xticks=range(4), xticklabels=["8K", "16K", "32K", "64K"],
               yticks=range(10), yticklabels=[f"qa{i}" for i in range(1, 11)],
               xlabel="Document length", ylabel="BABILong task")
        ax.tick_params(length=0, pad=4)
        for spine in ax.spines.values(): spine.set_visible(False)
        for y, x in zip(*np.where(np.isnan(z))):
            ax.text(x, y, "–", ha="center", va="center", fontsize=8, color="#687580")
    cbax = fig.add_axes([.85, .20, .019, .70])
    cb = fig.colorbar(im, cax=cbax, ticks=[-100, -50, 0, 50, 100])
    cb.set_label("Gain (percentage points)")
    cb.outline.set_visible(False)
    return fig


def scatter_figure(cases):
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.95))
    fig.subplots_adjust(left=.10, right=.98, bottom=.25, top=.84, wspace=.48)
    for ax, metric, scale, limit, unit in zip(axes, ("recall", "context_chars"),
            (100, .001), ((-9, 109), (-.5, 16)), ("Support recall (%)", "Context length (K chars)")):
        rows = rows_for(cases, metric)
        if metric == "recall":
            counts = collections.Counter((c["vanilla"][metric]*scale, c["query_specific"][metric]*scale) for c in rows)
            points = [(x, y, 9 + n*.65) for (x, y), n in sorted(counts.items())]
        else:
            points = [(c["vanilla"][metric]*scale, c["query_specific"][metric]*scale, 9) for c in rows]
        for status, color, marker in (("improve", BLUE, "o"), ("regress", ORANGE, "x"), ("tie", "#86949D", "s")):
            subset = []
            for x, y, area in points:
                gain = (y-x) if metric == "recall" else (x-y)
                label = "tie" if abs(gain) < 1e-10 else "improve" if gain > 0 else "regress"
                if status == label: subset.append((x, y, area))
            if subset:
                x, y, size = zip(*subset)
                ax.scatter(x, y, s=size, color=color, marker=marker, alpha=.65,
                           linewidths=.65 if marker == "x" else .3, clip_on=False)
        ax.plot(limit, limit, color="#9AA6AD", ls="--", lw=.8, zorder=0)
        ax.set(xlim=limit, ylim=limit, xlabel=f"Query-agnostic\n{unit}",
               ylabel=f"Query-specific\n{unit}", aspect="equal")
        if metric == "recall": ax.set(xticks=[0, 50, 100], yticks=[0, 50, 100])
        else: ax.set(xticks=[0, 8, 16], yticks=[0, 8, 16])
        axis(ax)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", ms=4, label="Higher recall / shorter context"),
                        Line2D([], [], color=ORANGE, marker="x", ls="", ms=4, label="Reverse"),
                        Line2D([], [], color="#86949D", marker="s", ls="", ms=4, label="Equal")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0), columnspacing=1.2)
    return fig


def initialize_templates(root):
    path = root/"data/babilong_results.json"
    if not path.exists():
        methods = ["full_context_icl", "agnostic_vanilla_sd", "agnostic_selective_sd",
                   "specific_vanilla_sd", "specific_selective_sd"]
        write(path, {"schema": "contextual-cartridges-babilong-paper-v1", "status": "pending",
             "protocol": "1000 paired requests per backbone; qa1-qa10; 250 at each length; no partial accuracy treated as final",
             "rows": [{"model": m, "method": a, "length": l, "n_expected": 250,
                       "n_completed": 0, "correct": None, "accuracy_percent": None,
                       "status": "pending", "source": None}
                      for m in MODELS for a in methods for l in LENGTHS]})
    path = root/"data/cache_results.json"
    if not path.exists():
        write(path, {"schema": "contextual-cartridges-cache-paper-v1", "status": "pending",
             "metric_note": "LongHealth accuracy; MTOB KE corpus chrF; QASPER mean gold-answer log-perplexity. Values are not interchangeable.",
             "rows": [{"model": m, "dataset": d, "method": "Contextual Cartridges", "cache_tokens": k,
                       "kv_bytes": None, "metric": METRICS[d][1], "direction": METRICS[d][2],
                       "value": None, "n_completed": 0, "paired_case_set_sha256": None,
                       "status": "pending", "source": None}
                      for m in MODELS for d in DATASETS for k in CACHES]})


def cache_figure(data, models=MODELS):
    fig, axes = plt.subplots(len(models), 3, figsize=(6.30, 1.93*len(models)+.38), squeeze=False)
    fig.subplots_adjust(left=.11, right=.98, bottom=.12 if len(models)>1 else .25,
                        top=.90 if len(models)>1 else .76, hspace=.80, wspace=.57)
    for i, model in enumerate(models):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            rows = sorted([r for r in data["rows"] if r["model"] == model and r["dataset"] == dataset],
                          key=lambda r:r["cache_tokens"])
            complete = [r for r in rows if r["status"] == "complete" and r["value"] is not None]
            if complete:
                hashes = {r["paired_case_set_sha256"] for r in complete}
                if None in hashes or len(hashes) != 1:
                    raise ValueError("Cache points must share an identified paired evaluation set")
                values = [r["value"] if r in complete else np.nan for r in rows]
                ax.plot([r["cache_tokens"] for r in rows], values, color=BLUE, marker="D", lw=1.5, ms=4)
            else:
                ax.text(.5, .52, "Pending", transform=ax.transAxes, ha="center", va="center", color="#86949D")
                ax.set_yticks([])
            ax.set_xscale("log", base=2)
            ax.set(xticks=CACHES, xticklabels=["512", "1K", "2K", "4K"], xlim=(440, 4800),
                   xlabel="KV positions", ylabel=METRICS[dataset][1])
            ax.set_title(f"{model}\n{METRICS[dataset][0]}", fontsize=8, pad=5)
            if dataset != "qasper": ax.set_ylim(0, 100)
            axis(ax)
    fig.legend(handles=[Line2D([],[],color=BLUE,marker="D",ms=4,lw=1.5,label="Contextual Cartridges")],
               loc="upper center", frameon=False)
    return fig


def save(fig, root, stem, manifest):
    outputs = []
    for ext in ("pdf", "svg", "png"):
        path = root/"figures"/f"{stem}.{ext}"
        fig.savefig(path, dpi=400, facecolor="white", transparent=False)
        outputs.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with Image.open(root/"figures"/f"{stem}.png") as im:
        pixels = list(im.size)
        dpi = list(im.info["dpi"])
    manifest.append({"name": stem, "size_inches": fig.get_size_inches().tolist(),
                     "png_pixels": pixels, "png_dpi": dpi, "outputs": outputs})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root=parser.parse_args().root.resolve()
    source=read(root/"data/synthetic_source.json"); cases=source["cases"]
    assert len(cases)==len({c["case"] for c in cases})==1000
    assert all(c["vanilla"]["questions"]==c["query_specific"]["questions"] for c in cases)
    assert all(sum(c[a]["questions"] for c in cases)==26451 for a in ARMS)
    assert collections.Counter(c["length"] for c in cases)=={l:250 for l in LENGTHS}
    assert collections.Counter(c["task"] for c in cases)=={f"qa{i}":100 for i in range(1,11)}
    # Fixed axes must not silently clip any observed point.
    assert all(0<=c[a]["context_chars"]*.001<=16 for c in cases for a in ARMS)
    initialize_templates(root)
    summaries={metric:stats(cases,metric,multi=(metric=="joint")) for metric in
               ("no_support","recall","joint","context_chars","question_words")}
    write(root/"data/synthetic_summary.json", {"n_requests":1000,"questions_per_arm":26451,
          "estimator":"paired equal-weight request means; joint restricted to multi-fact cases",
          "uncertainty":"none displayed", "metrics":summaries})
    with (root/"data/synthetic_per_request.csv").open("w",newline="") as f:
        keys=["case","task","length","support_fact_count","multi_fact"]
        fields=keys+[f"{a}_{k}" for a in ARMS for k in cases[0][a]]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for c in cases: writer.writerow({**{k:c[k] for k in keys},**{f"{a}_{k}":v for a in ARMS for k,v in c[a].items()}})
    with (root/"data/heatmap_cells.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["metric","task","length","n","vanilla","query_specific","gain_pp"]);writer.writeheader()
        for metric in ("no_support","joint"):
            for task in range(1,11):
                for length in LENGTHS:
                    s=stats(cases,metric,length=length,task=f"qa{task}",multi=metric=="joint")
                    writer.writerow({"metric":metric,"task":f"qa{task}","length":length,
                         **{k:s[k] for k in ("n","vanilla","query_specific")},
                         "gain_pp":None if s["delta"] is None else 100*s["delta"]*(-1 if metric=="no_support" else 1)})
    manifest=[]
    with mpl.rc_context(STYLE):
        for function, stem in [(curve_figure,"figure1_dilution_fragmentation"),
                               (heatmap_figure,"figure2_task_length"),
                               (scatter_figure,"figure3_paired_coverage_footprint")]:
            save(function(cases),root,stem,manifest)
        cache=read(root/"data/cache_results.json")
        save(cache_figure(cache),root,"figure4_cache_grid_pending",manifest)
        for model, slug in zip(MODELS,("qwen3_4b","olmo3_7b","gpt_oss_20b")):
            save(cache_figure(cache,(model,)),root,f"cache_{slug}_pending",manifest)
    write(root/"data/figure_manifest.json", {"source_sha256":hashlib.sha256((root/"data/synthetic_source.json").read_bytes()).hexdigest(),
          "palette":PALETTE,"uncertainty":"none", "aggregation":"case macro means",
          "scatter":"exact coordinates; recall point area = 9 + 0.65 * multiplicity in points squared; no jitter; axis padding preserves endpoint markers",
          "heatmap":"symmetric -100 to +100 pp; grey and dash = no eligible multi-fact requests",
          "missing":"six recall pairs omitted only for recall; all others preserved",
          "cache":"nulls are pending, no fabricated points; completed capacities require same paired-case hash",
          "matplotlib_version":mpl.__version__,"figures":manifest})
    print(json.dumps(summaries,indent=2))


if __name__=="__main__": main()
