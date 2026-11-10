"""Publication figures for measured query-specific synthetic-data properties.

No bars, confidence intervals, model-performance placeholders, smoothing, or
invented measurements. All contrasts preserve case pairing and matched exercise
budgets. Source JSON is converted to a portable, label-free plotting snapshot.

The provenance and visual review workflow follows Scientific Agent Skills:
https://arxiv.org/abs/2609.00065 (Kassis et al., 2026).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image


INK, DEEP, BLUE, CYAN, PALE = "#0B2A3A", "#1E4B6B", "#3180A8", "#6EC1D8", "#B7E3F1"
YELLOW, GOLD, ORANGE, RUST, BROWN = "#F7C948", "#F59E2C", "#E36A1C", "#D35B1D", "#8C3A1A"
GRAY, GRID = "#556B77", "#DBE7EC"
ARMS = ("vanilla", "query_specific")
LENGTHS = (8192, 16384, 32768, 65536)
TASKS = tuple(f"qa{i}" for i in range(1, 11))
PALETTE = [INK, DEEP, BLUE, CYAN, PALE, YELLOW, GOLD, ORANGE, RUST, BROWN]
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 11.5, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GRAY, "ytick.color": GRAY,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.edgecolor": GRID, "axes.linewidth": .8,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "contextual-cartridges-evidence-20261110",
    "axes.unicode_minus": True,
}
METRICS = {
    "no_support": ("support", "no_support_context_fraction"),
    "recall": ("support", "context_union_support_recall_min"),
    "joint": ("support", "complete_context_fraction_min"),
    "context_chars": ("support", "mean_evidence_characters"),
    "question_words": ("support", "mean_question_words"),
    "query_focus": ("local", "query_term_hit_fraction"),
    "distinct_answers": ("local", "distinct_answers"),
    "dominant_answer": ("local", "dominant_answer_fraction"),
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_snapshot(audit_path, annotation_path):
    audit = json.loads(Path(audit_path).read_text())
    annotations = {r["case"]: r for r in map(json.loads, Path(annotation_path).read_text().splitlines())}
    assert audit["schema"] == "query-anchored-dataset-audit-v1"
    assert audit["completed_cases"] == audit["planned_cases"] == 1000
    assert audit["benchmark_answers_read"] is False
    assert audit["dataset_gate_passed"] is True and not audit["issues"]
    cases = []
    for row in audit["cases"]:
        n_support = sum(bool(f.get("is_support")) for f in annotations[row["case"]]["facts"])
        item = {"case": row["case"], "task": row["task"], "length": row["length"],
                "support_fact_count": n_support, "multi_fact": n_support > 1}
        for source, arm in (("vanilla", "vanilla"), ("new_guided", "query_specific")):
            item[arm] = {k: row[source][group][name] for k, (group, name) in METRICS.items()}
            item[arm]["questions"] = row[source]["local"]["questions"]
        assert item["vanilla"]["questions"] == item["query_specific"]["questions"]
        cases.append(item)
    assert len({x["case"] for x in cases}) == 1000
    assert sum(x["multi_fact"] for x in cases) == 352
    assert sum(x["vanilla"]["questions"] for x in cases) == 26451
    # Reproduce published macro means before creating any new paired estimator.
    for source, arm in (("vanilla", "vanilla"), ("new_guided", "query_specific")):
        for metric, summary_key in (("no_support", "no_support_context_fraction_mean"),
                                    ("recall", "support_recall_lower_bound_mean"),
                                    ("joint", "all_support_context_fraction_lower_bound_mean")):
            values = [x[arm][metric] for x in cases if x[arm][metric] is not None]
            assert np.isclose(np.mean(values), audit["summaries"][source][summary_key], atol=1e-12)
    return {
        "schema": "contextual-cartridges-figure-snapshot-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen3-4B-Instruct-2507",
        "sources": [
            {"file": Path(audit_path).name, "sha256": digest(audit_path), "role": "per-case matched synthetic-data audit"},
            {"file": Path(annotation_path).name, "sha256": digest(annotation_path), "role": "post-generation support-count annotation"},
        ],
        "protocol": {
            "arms": {"vanilla": "Saved query-agnostic Cartridges synthetic conversations",
                     "query_specific": "Answer-guided, query-anchored v3 synthetic questions"},
            "n_cases": 1000, "questions_per_arm": 26451,
            "budget_matching": "Per case and per split, identical prefix counts from the source audit; no outcome-based filtering.",
            "estimator": "Equal-weight case mean; require both arms to be non-missing for each contrast.",
            "multi_fact_subset": "352 cases with more than one annotated supporting fact; not a strict cross-chunk-necessity test.",
            "missing": "Six vanilla support-recall values are missing; recall contrasts use 994 paired cases. Missing is never zero.",
            "semantic_correctness": "Not assessed by these dataset metrics.",
            "timing": "Context characters are an evidence-footprint measurement, not wall-clock time or GPU memory.",
            "uncertainty": "None displayed, as requested.",
        },
        "cases": sorted(cases, key=lambda x: x["case"]),
    }


def paired(cases, metric, *, length=None, task=None, multi=False):
    return [c for c in cases if (length is None or c["length"] == length)
            and (task is None or c["task"] == task) and (not multi or c["multi_fact"])
            and all(c[a][metric] is not None for a in ARMS)]


def means(cases, metric, **kwargs):
    rows = paired(cases, metric, **kwargs)
    return len(rows), [float(np.mean([r[a][metric] for r in rows])) for a in ARMS] if rows else [None, None]


def header(fig, number, title, subtitle):
    fig.text(.055, .952, f"CONTEXTUAL CARTRIDGES   /   {number}", fontsize=9.5, weight="bold", color=DEEP)
    fig.text(.055, .893, title, fontsize=21, weight="bold")
    fig.text(.055, .852, subtitle, fontsize=10.5, color=GRAY)
    # A fine visual key to the requested palette, outside the data area.
    for i, color in enumerate([INK, DEEP, BLUE, CYAN, PALE, YELLOW, GOLD, ORANGE, RUST, BROWN]):
        fig.add_artist(Line2D([.81 + i*.013, .821 + i*.013], [.957, .957],
                              color=color, lw=3, transform=fig.transFigure, solid_capstyle="round"))


def methods_legend(fig, *, y=.795, x=.055):
    handles = [Line2D([], [], color=ORANGE, ls=(0, (4, 2)), marker="o", mfc="white", mec=RUST,
                      lw=2, ms=6, label="Vanilla Cartridges data"),
               Line2D([], [], color=BLUE, marker="D", mfc=BLUE, mec=DEEP,
                      lw=2, ms=5.5, label="Query-specific data (ours)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x, y),
               frameon=False, ncol=2, columnspacing=2.5, handlelength=2.7, fontsize=10.5,
               borderaxespad=0)


def axis_style(ax, *, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.grid(axis=grid, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)


def panel_head(fig, x, y, letter, title, value, detail):
    fig.text(x, y, letter, color=BLUE, weight="bold", fontsize=11)
    fig.text(x+.022, y, title, weight="bold", fontsize=12)
    fig.text(x, y-.052, value, weight="bold", fontsize=20, color=DEEP)
    fig.text(x, y-.087, detail, fontsize=10, color=GRAY)


def length_curve(ax, cases, metric, *, multi=False, ylim=(0, 100), yticks=None):
    values = np.array([means(cases, metric, length=l, multi=multi)[1] for l in LENGTHS])*100
    x = np.arange(4)
    ax.fill_between(x, values[:, 0], values[:, 1], color=PALE, alpha=.38, zorder=1)
    for i, color, marker, linestyle in ((0, ORANGE, "o", (0, (4, 2))), (1, BLUE, "D", "-")):
        ax.plot(x, values[:, i], color=color, marker=marker, ls=linestyle, lw=2.3, ms=6.7,
                mfc="white" if i == 0 else color, mec=RUST if i == 0 else DEEP, mew=1.1, zorder=4)
        # Values at all four measured lengths; line segments only connect measured points.
        for j, v in enumerate(values[:, i]):
            # Outer labels keep the small third-panel separation legible.
            offset = 10 if i == 1 else -15
            if metric == "no_support": offset = -16 if i == 1 else 9
            if metric == "joint" and i == 0: offset = 8 if v < 2 else -15
            ax.annotate(f"{v:.1f}", (j, v), xytext=(0, offset), textcoords="offset points",
                        ha="center", fontsize=10, color=DEEP if i == 1 else BROWN)
    ax.set(xlim=(-.2, 3.2), ylim=ylim, xticks=x, xticklabels=["8K", "16K", "32K", "64K"])
    if yticks is not None: ax.set_yticks(yticks)
    ax.set_xlabel("Document length")
    axis_style(ax)


def figure1(cases):
    fig = plt.figure(figsize=(13.2, 6.1))
    header(fig, "01", "The same practice budget, more relevant evidence",
           "BABILong · Qwen3-4B · 1,000 document–query pairs · 26,451 matched synthetic questions per method")
    methods_legend(fig, y=.786)
    panels = [(.070, "a", "DILUTION", "no_support", False),
              (.391, "b", "SUPPORT COVERAGE", "recall", False),
              (.712, "c", "FRAGMENTATION", "joint", True)]
    for x, letter, title, metric, multi in panels:
        n, (v, q) = means(cases, metric, multi=multi)
        value = f"{100*v:.1f} → {100*q:.1f}%"
        detail = ({"no_support": f"{100*(v-q):.1f} pp less zero-support context",
                   "recall": f"{100*(q-v):+.1f} pp support-fact recall",
                   "joint": f"{q/v:.2f}× all-fact co-exposure"})[metric]
        panel_head(fig, x, .726, letter, title, value, detail)
        ax = fig.add_axes([x, .213, .244, .355])
        if metric == "joint":
            length_curve(ax, cases, metric, multi=True, ylim=(-1, 25), yticks=[0, 5, 10, 15, 20, 25])
            ax.set_ylabel("Exercise contexts with all support (%)")
            foot = "352 multi-fact cases · 88 per length"
        elif metric == "recall":
            length_curve(ax, cases, metric, ylim=(0, 108), yticks=[0, 25, 50, 75, 100])
            ax.set_ylabel("Support facts covered by dataset (%)")
            foot = "994 paired cases · 6 missing excluded"
        else:
            length_curve(ax, cases, metric, ylim=(0, 108), yticks=[0, 25, 50, 75, 100])
            ax.set_ylabel("Exercise contexts with zero support (%)")
            foot = "1,000 paired cases · 250 per length"
        fig.text(x, .105, foot, fontsize=9.5, color=GRAY)
    fig.text(.055, .040, "Equal-weight case means. Shading marks the gap between methods. Coverage uses post-generation supporting-fact annotations.",
             fontsize=9.3, color=GRAY)
    return fig


def gains_grid(cases, metric, multi=False):
    z = np.full((10, 4), np.nan)
    n = np.zeros((10, 4), int)
    for i, t in enumerate(TASKS):
        for j, l in enumerate(LENGTHS):
            n[i, j], (v, q) = means(cases, metric, length=l, task=t, multi=multi)
            if n[i, j]: z[i, j] = 100*(v-q if metric == "no_support" else q-v)
    return z, n


def figure2(cases):
    fig = plt.figure(figsize=(13.2, 7.4))
    header(fig, "02", "Inspecting every task and document length",
           "Paired gains over vanilla synthetic data · percentage points · all 40 task–length groups retained")
    cmap = LinearSegmentedColormap.from_list("evidence_gain", [BROWN, ORANGE, YELLOW, "#FFFFFF", PALE, BLUE, INK])
    norm = Normalize(-100, 100)
    for x, letter, title, metric, multi in [(.075, "a", "LESS DILUTION", "no_support", False),
                                           (.395, "b", "HIGHER COVERAGE", "recall", False),
                                           (.715, "c", "MORE JOINT EXPOSURE", "joint", True)]:
        z, count = gains_grid(cases, metric, multi)
        positive, available = int(np.sum(z > 0)), int(np.sum(~np.isnan(z)))
        fig.text(x, .786, letter, weight="bold", color=BLUE, fontsize=11)
        fig.text(x+.022, .786, title, weight="bold", fontsize=11.7)
        fig.text(x, .748, f"{positive} / {available} groups improve", fontsize=16, weight="bold", color=DEEP)
        ax = fig.add_axes([x, .251, .235, .455])
        ax.set(xlim=(-.5, 3.5), ylim=(9.5, -.5), xticks=range(4), yticks=range(10),
               xticklabels=["8K", "16K", "32K", "64K"], yticklabels=[t.upper() for t in TASKS])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=6, labelsize=10.2)
        for sp in ax.spines.values(): sp.set_visible(False)
        for i in range(10):
            for j in range(4):
                value = z[i, j]
                face = "#F2F5F6" if np.isnan(value) else cmap(norm(value))
                ax.add_patch(FancyBboxPatch((j-.475, i-.455), .95, .91,
                             boxstyle="round,pad=0,rounding_size=.08", fc=face, ec="none"))
                if np.isnan(value):
                    label, color = "—", GRAY
                else:
                    label = f"{value:+.1f}" if abs(value) >= .05 else "0.0"
                    dark_text = BROWN if value < 0 else INK
                    color = (dark_text if contrast(dark_text, face) >= 4.5
                             else "white" if contrast("white", face) >= 4.5 else "#000000")
                    assert contrast(color, face) >= 4.5
                ax.text(j, i, label, ha="center", va="center", fontsize=10.4,
                        weight="bold" if not np.isnan(value) else "normal", color=color)
        text = {"no_support": "Vanilla − query-specific\nZero-support context share",
                "recall": "Query-specific − vanilla\nDataset support-fact recall",
                "joint": "Query-specific − vanilla\nAll-fact co-exposure; multi-fact cases"}[metric]
        fig.text(x, .198, text, fontsize=10, color=GRAY, linespacing=1.6)
    cax = fig.add_axes([.31, .105, .38, .017])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[-100, -50, 0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    fig.text(.27, .109, "Worse", ha="right", va="center", color=BROWN, fontsize=10)
    fig.text(.73, .109, "Better", va="center", color=DEEP, fontsize=10)
    fig.text(.055, .034, "Each cell averages its paired cases; maximum n = 25. “—” = no multi-fact cases, not zero gain. Exact counts are in the source table.",
             fontsize=9.3, color=GRAY)
    return fig


def ecdf(ax, values, *, color, ls, marker):
    x, counts = np.unique(np.array(values, float), return_counts=True)
    y = np.cumsum(counts) / len(values)*100
    ax.step(np.r_[0, x], np.r_[0, y], where="post", color=color, ls=ls, lw=2.5)
    for quantile in [.25, .5, .75]:
        idx = np.searchsorted(y, quantile*100)
        ax.plot(x[idx], y[idx], marker=marker, ms=6, color=color,
                mfc="white" if marker == "o" else color, zorder=5)


def figure3(cases):
    fig = plt.figure(figsize=(13.2, 6.5))
    header(fig, "03", "More coverage from a smaller evidence footprint",
           "BABILong · Matched question budgets · Saved evidence contexts and annotated supporting-fact coverage")
    methods_legend(fig, y=.786)
    _, (v, q) = means(cases, "context_chars")
    panel_head(fig, .070, .724, "a", "COVERAGE × CONTEXT SIZE", f"{v/1000:.2f}K → {q/1000:.2f}K characters",
               f"{100*(1-q/v):.1f}% smaller mean context · identical exercise counts")
    panel_head(fig, .691, .724, "b", "ALL 1,000 CASES", "The full size distribution",
               "Empirical cumulative curves · no smoothing")
    ax = fig.add_axes([.070, .215, .495, .358])
    rows = paired(cases, "recall")
    for arm, color, marker in (("vanilla", ORANGE, "o"), ("query_specific", BLUE, "D")):
        for task in TASKS:
            for length in LENGTHS:
                rs = [r for r in rows if r["task"] == task and r["length"] == length]
                xx = np.mean([r[arm]["context_chars"] for r in rs])/1000
                yy = np.mean([r[arm]["recall"] for r in rs])*100
                ax.scatter(xx, yy, s=33, marker=marker, facecolor="white" if arm == "vanilla" else CYAN,
                           edgecolor=RUST if arm == "vanilla" else BLUE, lw=.8, alpha=.78, zorder=3)
    for idx, length in enumerate(LENGTHS):
        rs = [r for r in rows if r["length"] == length]
        xs = [np.mean([r[a]["context_chars"] for r in rs])/1000 for a in ARMS]
        ys = [np.mean([r[a]["recall"] for r in rs])*100 for a in ARMS]
        ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=12, lw=1.25, color=BLUE,
                                    alpha=.62, shrinkA=6, shrinkB=7, connectionstyle=f"arc3,rad={.03*(idx-1.5)}"), zorder=2)
        for i, marker in enumerate(("o", "D")):
            ax.scatter(xs[i], ys[i], marker=marker, s=85, fc="white" if i == 0 else BLUE,
                       ec=BROWN if i == 0 else INK, lw=1.1, zorder=6)
        ax.annotate(f"{length//1024}K", (xs[0], ys[0]), xytext=(11, -1), textcoords="offset points",
                    color=BROWN, fontsize=10, va="center", weight="bold")
    ax.set(xlim=(0, 11), ylim=(0, 108), xticks=[0, 2, 4, 6, 8, 10], yticks=[0, 25, 50, 75, 100],
           xlabel="Mean evidence context per question (thousand characters)",
           ylabel="Dataset support-fact recall (%)")
    axis_style(ax, grid="both")
    bx = fig.add_axes([.691, .215, .255, .358])
    for arm, color, marker, ls in (("vanilla", ORANGE, "o", (0, (4, 2))),
                                   ("query_specific", BLUE, "D", "-")):
        vals = [r[arm]["context_chars"]/1000 for r in paired(cases, "context_chars")]
        ecdf(bx, vals, color=color, ls=ls, marker=marker)
    maximum = max(r[a]["context_chars"] for r in cases for a in ARMS)/1000
    xmax = max(10, np.ceil(maximum))
    bx.set(xlim=(0, xmax), ylim=(0, 105), yticks=[0, 25, 50, 75, 100],
           xlabel="Evidence context (thousand characters)", ylabel="Cases at or below this size (%)")
    axis_style(bx)
    fig.text(.070, .105, "Small points: all 40 task–length means. Large points / arrows: four length means on 994 paired recall cases.",
             fontsize=9.3, color=GRAY)
    fig.text(.055, .042, "Right: mean evidence size within each case, then the empirical distribution over 1,000 cases. Markers locate its 25th, 50th and 75th percentiles.",
             fontsize=9.3, color=GRAY)
    return fig


def write_tables(cases, out):
    records = []
    for c in cases:
        for a in ARMS:
            records.append({**{k: c[k] for k in ("case", "task", "length", "support_fact_count", "multi_fact")},
                            "method": a, **c[a]})
    with (out / "per_case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    summary = []
    for metric in ("no_support", "recall", "joint", "context_chars", "query_focus", "question_words"):
        for scope in ("all", *LENGTHS):
            kwargs = {} if scope == "all" else {"length": scope}
            if metric == "joint": kwargs["multi"] = True
            n, (v, q) = means(cases, metric, **kwargs)
            multiplier = 1 if metric in ("context_chars", "question_words") else 100
            summary.append({"metric": metric, "scope": scope, "multi_fact_only": metric == "joint", "n_paired": n,
                            "vanilla": v*multiplier, "query_specific": q*multiplier, "delta": (q-v)*multiplier,
                            "unit": "characters" if metric == "context_chars" else "words" if metric == "question_words" else "percent"})
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n"); w.writeheader(); w.writerows(summary)
    cell_rows = []
    for metric, multi in (("no_support", False), ("recall", False), ("joint", True)):
        z, n = gains_grid(cases, metric, multi)
        for i, task in enumerate(TASKS):
            for j, length in enumerate(LENGTHS):
                cell_rows.append({"metric": metric, "task": task, "length": length, "n_paired": int(n[i, j]),
                                  "gain_pp": None if np.isnan(z[i,j]) else float(z[i,j])})
    with (out / "task_length_gains.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(cell_rows)
    return summary


def contrast(foreground, background="#FFFFFF"):
    def lum(color):
        a = np.array(mpl.colors.to_rgb(color))
        a = np.where(a <= .04045, a/12.92, ((a+.055)/1.055)**2.4)
        return float(a @ np.array([.2126, .7152, .0722]))
    a, b = sorted([lum(foreground), lum(background)])
    return (b+.05)/(a+.05)


def export(fig, out, name, description):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    # Inspect actual rendered text extents; exceptions include intentionally hidden tick text.
    clipped = []
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > page.x1+1 or box.y1 > page.y1+1:
                clipped.append(text.get_text())
    if clipped: raise ValueError(f"Text outside page for {name}: {clipped}")
    files = []
    for suffix in ("png", "pdf", "svg"):
        p = out / f"{name}.{suffix}"
        kwargs = {"dpi": 300, "facecolor": "white", "bbox_inches": None}
        if suffix == "png": kwargs["metadata"] = {"Title": description, "Software": "Matplotlib"}
        if suffix == "pdf": kwargs["metadata"] = {"Title": description, "Author": "Contextual Cartridges"}
        if suffix == "svg": kwargs["metadata"] = {"Title": description, "Description": description}
        fig.savefig(p, **kwargs)
        if suffix == "svg":
            p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
        files.append({"file": p.name, "sha256": digest(p), "bytes": p.stat().st_size})
    p = out / f"{name}.png"
    with Image.open(p) as im:
        raster = {"pixels": list(im.size), "dpi": list(im.info.get("dpi", ())), "mode": im.mode,
                  "opaque": im.convert("RGBA").getchannel("A").getextrema() == (255,255)}
    record = {"name": name, "description": description, "physical_inches": list(fig.get_size_inches()),
              "text_outside_page": clipped, "raster": raster, "outputs": files}
    dump(out / f"{name}.export.json", record)
    plt.close(fig)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    if args.data:
        data = json.loads(args.data.read_text())
    else:
        if not args.audit or not args.annotations: p.error("Provide --data, or both --audit and --annotations")
        data = make_snapshot(args.audit, args.annotations)
    assert data["schema"] == "contextual-cartridges-figure-snapshot-v1"
    cases = data["cases"]
    assert len(cases) == 1000 and sum(c["multi_fact"] for c in cases) == 352
    assert all(c["vanilla"]["questions"] == c["query_specific"]["questions"] for c in cases)
    dump(out / "source_data.json", data)
    summary = write_tables(cases, out)
    with mpl.rc_context(STYLE):
        figures = [
            export(figure1(cases), out, "figure1_dilution_fragmentation", "Matched data: less zero-support context, higher support recall, and more all-fact co-exposure."),
            export(figure2(cases), out, "figure2_task_length", "All task-by-length gains over vanilla; missing subgroups distinguished from zero."),
            export(figure3(cases), out, "figure3_evidence_efficiency", "Support coverage and evidence-context size, with all-case empirical context-size distributions."),
        ]
    audit = {"palette": PALETTE, "contrast_against_white": {c: contrast(c) for c in PALETTE},
             "text_colors": [INK, DEEP, GRAY, BROWN],
             "text_color_contrast": {c: contrast(c) for c in [INK, DEEP, GRAY, BROWN]},
             "redundant_encoding": "Orange dashed/open circles; blue solid/diamonds; signed numeric values in heatmap.",
             "caveat": "Pale colors are fills, not body text. Heatmap labels use neutral black only where neither navy nor white reaches 4.5:1. Contrast screening is not accessibility certification."}
    assert all(contrast(c) >= 4.5 for c in [INK, DEEP, GRAY, BROWN])
    dump(out / "palette_audit.json", audit)
    dump(out / "figure_manifest.json", {
        "source_sha256": digest(out / "source_data.json"), "script_sha256": digest(__file__),
        "sources": data["sources"], "figures": figures, "summary": summary,
        "method": data["protocol"], "style": "User-requested blue/cyan and yellow/orange; no bars or confidence intervals.",
        "aggregation": "Paired equal-case means; no weighting by number of generated exercises.",
        "heatmap": "Signed percentage-point gains; fixed symmetric range [-100,100]; every task-length group retained.",
        "ecdf": "Exact empirical steps; no smoothing, density estimation, interpolation, or resampling.",
        "destination": "General research report and editable figure assets; no publisher compliance claim.",
        "versions": {"matplotlib": mpl.__version__, "numpy": np.__version__},
    })
    print(json.dumps({"output_dir": str(out), "figures": [r["name"] for r in figures], "paired_recall_cases": len(paired(cases, "recall"))}))


if __name__ == "__main__":
    main()
