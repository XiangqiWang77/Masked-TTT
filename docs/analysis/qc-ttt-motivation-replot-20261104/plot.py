#!/usr/bin/env python3
"""Three publication-style figures from the frozen group-level snapshots."""
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
RUST, GOLD, DARK = "#6F1A07", "#A8763E", "#2B2118"
IVORY, GRID = "#F7F3E3", "#EAE5E0"
PALE, MUTED = "#DBCEC0", "#75695F"


def csv_rows(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def export(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=600, facecolor="white")
    svg = OUT / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    png = OUT / f"{stem}.png"
    with Image.open(png) as im:
        if im.mode == "RGBA" and im.getchannel("A").getextrema() != (255, 255):
            raise ValueError("PNG must have an opaque white background")
        im.convert("RGB").save(png, dpi=(600, 600))
    plt.close(fig)


def style_axis(ax):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(DARK)
        ax.spines[spine].set_linewidth(.7)
    ax.tick_params(colors=DARK, labelsize=8.1, length=3, width=.7)
    ax.set_axisbelow(True)


def small_cells(ax, x, y, ncol, nrow, highlighted, sx=.019, sy=.025, gap=.004):
    for row in range(nrow):
        for col in range(ncol):
            i = row * ncol + col
            color = RUST if i in highlighted else PALE
            ax.add_patch(Rectangle((x + col * (sx + gap), y + row * (sy + gap)),
                                   sx, sy, facecolor=color, edgecolor="none", alpha=.88))


def figure_one(summary):
    """Conceptual two-view diagram above actual OLMo performance."""
    fig = plt.figure(figsize=(4.8, 4.8), facecolor="white")
    ax = fig.add_axes((.08, .48, .84, .49))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.23, .98, "Full KV", ha="center", va="top", color=DARK, fontsize=10, fontweight="bold")
    ax.text(.77, .98, "Selected KV", ha="center", va="top", color=DARK, fontsize=10, fontweight="bold")
    ax.text(.23, .89, "global view · dilution risk", ha="center", color=DARK, fontsize=7.4)
    ax.text(.77, .89, "focused · fragmentation risk", ha="center", color=DARK, fontsize=7.4)
    small_cells(ax, .075, .73, 14, 3, {3, 18, 37, 41}, sx=.018, sy=.024, gap=.004)
    # Three separated groups depict selected spans and the gaps between them.
    for i, x in enumerate((.625, .735, .845)):
        small_cells(ax, x, .76, 3, 2, {0, 4} if i != 1 else {2}, sx=.020, sy=.025, gap=.005)
    for x in (.71, .82):
        ax.plot((x, x+.015), (.79, .79), color=MUTED, lw=1.0, linestyle=(0, (1, 1.5)))
    ax.text(.23, .69, "many competing positions", ha="center", color=MUTED, fontsize=7.1)
    ax.text(.77, .69, "incomplete facts: 37.5%*", ha="center", color=MUTED, fontsize=7.1)
    for start, end, rad in (((.25, .62), (.43, .39), .18), ((.75, .62), (.57, .39), -.18)):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                     connectionstyle=f"arc3,rad={rad}", lw=1.05, color=GOLD))
    ax.text(.18, .51, "full-context\nadaptation", ha="center", va="center", color=DARK, fontsize=7.0)
    ax.text(.82, .51, "selected ICL\nsupervision", ha="center", va="center", color=DARK, fontsize=7.0)
    ax.add_patch(FancyBboxPatch((.365, .16), .27, .22, boxstyle="round,pad=.012,rounding_size=.025",
                               facecolor=IVORY, edgecolor=RUST, linewidth=1.0))
    ax.text(.5, .305, "QC-TTT", ha="center", va="center", color=RUST, fontsize=10, fontweight="bold")
    ax.text(.5, .23, "adapted compact KV", ha="center", va="center", color=DARK, fontsize=7.0)

    bottom = fig.add_axes((.16, .12, .79, .30))
    style_axis(bottom)
    bottom.grid(axis="y", color=GRID, linewidth=.55)
    olmo = summary["olmo"]
    for length in (8192, 16384, 32768, 65536):
        row = olmo["by_context"][str(length)]
        vals = [100*row[name]/row["n"] for name in ("full_correct", "selected_correct", "qc_correct")]
        bottom.plot((0, 1, 2), vals, color=GOLD, alpha=.33, lw=.8, marker="o", markersize=2.1, zorder=1)
    vals = [100*olmo[name]/olmo["n"] for name in ("full_correct", "selected_correct", "qc_correct")]
    bottom.plot((0, 1, 2), vals, color=RUST, lw=1.5, marker="o", markersize=4.2, zorder=3)
    for x, value, yoff in zip((0, 1, 2), vals, (4, 5, 5)):
        bottom.annotate(f"{value:.1f}", (x, value), xytext=(0, yoff), textcoords="offset points",
                        ha="center", color=RUST, fontsize=8.2, fontweight="bold")
    bottom.set(xlim=(-.17, 2.18), ylim=(25, 55), ylabel="Answer accuracy (%)")
    bottom.set_xticks((0, 1, 2), ("Full KV", "Selected ICL", "QC-TTT"))
    bottom.set_yticks((30, 40, 50))
    bottom.tick_params(axis="x", length=0, pad=6)
    bottom.yaxis.label.set_size(8.2)
    bottom.text(.02, .96, "OLMo · BABILONG", transform=bottom.transAxes, color=MUTED,
                fontsize=7.1, va="top")
    export(fig, "motivation_1_views_and_qc")


def figure_two(rows, summary):
    """Half-violin + exact-y beeswarm, with paired overall change marked."""
    fig, ax = plt.subplots(figsize=(4.8, 4.8), facecolor="white")
    fig.subplots_adjust(left=.19, right=.97, bottom=.18, top=.93)
    style_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.axhline(0, color="#9A8E84", linewidth=.9, linestyle=(0, (3, 3)), zorder=1)
    styles = (("kv_whole_ntp", "Whole NTP", DARK, "o"),
              ("kv_selected_ntp", "Selected NTP", GOLD, "s"),
              ("kv_sd_soft", "SD", RUST, "^") )
    support = np.linspace(-29, 37, 401)
    for i, (arm, name, color, marker) in enumerate(styles):
        group = [r for r in rows if r["arm"] == arm]
        if len(group) != 40:
            raise ValueError("Expected 40 task/length groups per method")
        ys = np.array([100*(int(r["gained"])-int(r["lost"]))/int(r["n"]) for r in group])
        # Gaussian kernel with bandwidth 4 pp. Only the categorical x coordinate is spread.
        density = np.exp(-.5*((support[:, None]-ys[None, :])/4.0)**2).mean(axis=1)
        width = .29*density/max(density)
        ax.fill_betweenx(support, i-width, i, facecolor=color, alpha=.14, linewidth=0, zorder=1)
        ax.plot(i-width, support, color=color, alpha=.55, lw=.8, zorder=2)
        used = Counter()
        for y in ys:
            level = round(float(y), 6)
            dx = .065 + .036*used[level]
            used[level] += 1
            ax.scatter(i+dx, y, s=13, marker=marker, color=color, alpha=.42,
                       edgecolors="white", linewidths=.25, zorder=3)
        totals = summary["qwen"]["methods"][arm]
        net = 100*(totals["gained"]-totals["lost"])/summary["qwen"]["n"]
        ax.scatter(i, net, s=54, marker="D", facecolor=color, edgecolor="white",
                   linewidth=.8, zorder=5)
        ax.annotate(f"{net:+.1f}", (i, net), xytext=(-3, 8 if net >= 0 else -11),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8.2, color=color, fontweight="bold")
    ax.set(xlim=(-.48, 2.48), ylim=(-29, 37), ylabel="Accuracy change vs vanilla (pp)")
    ax.set_xticks((0, 1, 2), ("Whole NTP", "Selected NTP", "SD"))
    ax.set_yticks((-20, -10, 0, 10, 20, 30))
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.label.set_size(8.5)
    ax.text(.02, .98, "Qwen3-4B · matched questions", transform=ax.transAxes,
            va="top", color=MUTED, fontsize=7.4)
    export(fig, "motivation_2_ntp_sd_raincloud")


def figure_three(rows, summary):
    """Paired accuracy and recorded KV footprint, overall and 40 real groups."""
    fig, ax = plt.subplots(figsize=(4.8, 4.8), facecolor="white")
    fig.subplots_adjust(left=.17, right=.95, bottom=.17, top=.90)
    style_axis(ax)
    ax.grid(color=GRID, linewidth=.55)
    for row in rows:
        n = int(row["n"])
        qc_x = 100*float(row["median_cartridge_full_ratio"])
        full_y = 100*int(row["full_correct"])/n
        qc_y = 100*int(row["qc_correct"])/n
        ax.plot((100, qc_x), (full_y, qc_y), color=GOLD, alpha=.10, lw=.55, zorder=1)
        ax.scatter([100], [full_y], s=11, facecolor=DARK, alpha=.35, linewidths=0, zorder=2)
        ax.scatter([qc_x], [qc_y], s=14, facecolor=RUST, alpha=.32, linewidths=0, zorder=2)
    s = summary["olmo"]
    full_y = 100*s["full_correct"]/s["n"]
    qc_y = 100*s["qc_correct"]/s["n"]
    qc_x = 100*s["median_cartridge_full_ratio"]
    ax.add_patch(FancyArrowPatch((100, full_y), (qc_x, qc_y), arrowstyle="-|>", mutation_scale=10,
                 color=RUST, lw=1.6, zorder=4, shrinkA=5, shrinkB=6))
    ax.scatter([100], [full_y], s=58, marker="o", facecolor=DARK, edgecolor="white", lw=.8, zorder=5)
    ax.scatter([qc_x], [qc_y], s=76, marker="D", facecolor=RUST, edgecolor="white", lw=.9, zorder=5)
    ax.annotate(f"{full_y:.1f}%", (100, full_y), xytext=(-2, -16), textcoords="offset points",
                ha="right", color=DARK, fontsize=8.5, fontweight="bold")
    ax.annotate(f"{qc_y:.1f}%", (qc_x, qc_y), xytext=(-3, 9), textcoords="offset points",
                ha="center", color=RUST, fontsize=8.5, fontweight="bold")
    ax.annotate("70.9% less KV", (qc_x, qc_y), xytext=(6, -13), textcoords="offset points",
                ha="left", color=RUST, fontsize=7.6)
    ax.set(xlim=(0, 107), ylim=(-2, 102), xlabel="Document state size (% of Full KV)",
           ylabel="Answer accuracy (%)")
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    legend = [Line2D([], [], marker="o", linestyle="", color=DARK, markersize=4.5, label="Full KV"),
              Line2D([], [], marker="D", linestyle="", color=RUST, markersize=4.5, label="QC-TTT · KV + adapter"),
              Line2D([], [], linestyle="-", color=GOLD, alpha=.45, linewidth=1.0,
                     label="Task × length groups")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.56, .97), ncol=3, frameon=False,
               fontsize=6.8, handlelength=.9, handletextpad=.3, columnspacing=.65)
    export(fig, "motivation_3_quality_storage")


def main():
    summary = json.loads((DATA / "summary.json").read_text())
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": False}):
        figure_one(summary)
        figure_two(csv_rows("qwen_objective_changes.csv"), summary)
        figure_three(csv_rows("olmo_task_length.csv"), summary)
    print("Rendered three square figures in PDF, SVG, and 600-dpi PNG.")


if __name__ == "__main__":
    main()
