#!/usr/bin/env python3
"""Render three compact motivation figures from the frozen local CSV files."""
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "figures"
DARK = "#2B2118"
CARAMEL = "#A8763E"
RUST = "#6F1A07"
GRID = "#E7E2DC"


def load(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def base(figsize=(3.65, 3.65)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.subplots_adjust(left=.20, right=.96, bottom=.18, top=.84)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color(DARK)
        ax.spines[name].set_linewidth(.8)
    ax.tick_params(axis="both", colors=DARK, labelsize=8, length=3, width=.75)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT / (stem + "." + suffix), dpi=600, facecolor="white", transparent=False)
    svg = OUTPUT / (stem + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    png = OUTPUT / (stem + ".png")
    with Image.open(png) as image:
        if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
            raise ValueError("Expected opaque figure")
        image.convert("RGB").save(png, dpi=(600, 600))
    plt.close(fig)


def coverage_figure():
    rows = load("coverage.csv")
    table = {(row["group"], row["method"]): row for row in rows}
    groups = ("0%", "1–99%", "100%")
    styles = {
        "whole": dict(label="Full KV", color=DARK, marker="o", ls="-", offset=-.04),
        "teacher": dict(label="Selected ICL", color=CARAMEL, marker="s", ls="--", offset=0),
        "cartridge": dict(label="Cartridge", color=RUST, marker="^", ls="-", offset=.04),
    }
    if len(table) != 9 or [int(table[g, "whole"]["n"]) for g in groups] != [148, 166, 524]:
        raise ValueError("The OLMo fact-coverage groups changed")
    fig, ax = base()
    ax.grid(axis="y", color=GRID, linewidth=.65)
    for name, style in styles.items():
        x = [i + style["offset"] for i in range(3)]
        y = [float(table[g, name]["accuracy_pct"]) for g in groups]
        ax.plot(x, y, color=style["color"], linestyle=style["ls"], linewidth=1.45,
                marker=style["marker"], markersize=5.4, markeredgewidth=1.05,
                markerfacecolor="white" if name == "teacher" else style["color"],
                markeredgecolor=style["color"], label=style["label"], zorder=3)
    ax.set(xlim=(-.2, 2.2), ylim=(0, 65), ylabel="Answer accuracy (%)",
           xlabel="Selected support fact coverage")
    ax.set_xticks(range(3), groups)
    ax.set_yticks(range(0, 61, 10))
    ax.xaxis.label.set_size(8.7)
    ax.yaxis.label.set_size(8.7)
    fig.legend(loc="upper center", bbox_to_anchor=(.57, .975), ncol=3,
               frameon=False, fontsize=7.15, handlelength=1.5, handletextpad=.35, columnspacing=.7)
    save(fig, "motivation_1_coverage_tradeoff")


def objectives_figure():
    rows = {row["arm"]: row for row in load("objective_accuracy.csv")}
    if set(rows) != {"vanilla", "kv_whole_ntp", "kv_selected_ntp", "kv_sd_soft"} or any(
        int(row["n"]) != 1000 for row in rows.values()):
        raise ValueError("The paired Qwen objective comparison changed")
    baseline = float(rows["vanilla"]["accuracy_pct"])
    arms = (("kv_whole_ntp", "Whole NTP", DARK, "o"),
            ("kv_selected_ntp", "Selected NTP", CARAMEL, "s"),
            ("kv_sd_soft", "SD", RUST, "^"))
    fig, ax = base()
    fig.subplots_adjust(left=.31, right=.94, bottom=.20, top=.83)
    ax.grid(axis="x", color=GRID, linewidth=.65)
    ax.axvline(baseline, color="#A9A19A", linewidth=.95, linestyle=(0, (3, 3)), zorder=1)
    for y, (arm, label, color, marker) in zip((2, 1, 0), arms):
        value = float(rows[arm]["accuracy_pct"])
        ax.plot([baseline, value], [y, y], color=color, alpha=.48, linewidth=1.4, zorder=2)
        ax.scatter([baseline], [y], s=27, marker="o", facecolor="white", edgecolor=DARK,
                   linewidth=.85, zorder=4)
        ax.scatter([value], [y], s=40, marker=marker, color=color, edgecolor="white",
                   linewidth=.45, zorder=5)
        offset, alignment = ((-5, 0), "right") if value < baseline else ((5, 0), "left")
        ax.annotate(f"{value:.1f}", (value, y), xytext=offset, textcoords="offset points",
                    ha=alignment, va="center", color=color, fontsize=7.7)
    ax.set_xlim(46, 61)
    ax.set_ylim(-.55, 2.55)
    ax.set_xticks((46, 50, 54, 58, 61))
    ax.set_yticks((2, 1, 0), [item[1] for item in arms])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel("Answer accuracy (%) · Qwen3-4B", fontsize=8.4, labelpad=7)
    legend = [Line2D([], [], marker="o", linestyle="", markersize=4.8, markerfacecolor="white",
                     markeredgecolor=DARK, label=f"Before: {baseline:.1f}%"),
              Line2D([], [], marker="o", linestyle="", markersize=4.8, markerfacecolor=RUST,
                     markeredgecolor=RUST, label="After: ≤16 updates")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.57, .975), ncol=2,
               frameon=False, fontsize=7.2, handlelength=1, columnspacing=1.15)
    save(fig, "motivation_2_ntp_vs_sd")


def storage_figure():
    rows = load("storage_points.csv")
    if len(rows) != 1000:
        raise ValueError("Expected 1,000 recorded OLMo KV sizes")
    full = [int(row["full_bytes"]) / 2**30 for row in rows]
    compact = [int(row["cartridge_bytes"]) / 2**30 for row in rows]
    fig, ax = base()
    fig.subplots_adjust(left=.20, right=.95, bottom=.18, top=.84)
    ax.grid(color=GRID, linewidth=.6)
    ax.scatter(full, compact, s=5.0, color=CARAMEL, alpha=.22, linewidths=0, rasterized=True, zorder=2)
    ax.plot([0, 10], [0, 10], color="#9C9690", linestyle=(0, (3, 3)), linewidth=.9, zorder=1)
    for length in (8192, 16384, 32768, 65536):
        subset = [row for row in rows if int(row["context_tokens"]) == length]
        if len(subset) != 250:
            raise ValueError("Expected 250 OLMo examples per context length")
        x = statistics.median(int(row["full_bytes"]) / 2**30 for row in subset)
        y = statistics.median(int(row["cartridge_bytes"]) / 2**30 for row in subset)
        ax.scatter([x], [y], s=30, color=RUST, marker="D", edgecolor="white", linewidth=.45, zorder=4)
        ax.annotate(str(length // 1024) + "K", (x, y), xytext=(4, 3), textcoords="offset points",
                    color=RUST, fontsize=7.0)
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="Full document KV (GiB)",
           ylabel="Cartridge KV + adapter (GiB)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((0, 2, 4, 6, 8, 10))
    ax.set_yticks((0, 2, 4, 6, 8, 10))
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    legend = [Line2D([], [], marker="o", linestyle="", markersize=3.8, color=CARAMEL,
                     label="Individual requests"),
              Line2D([], [], marker="D", linestyle="", markersize=4, color=RUST, label="Median"),
              Line2D([], [], linestyle=(0, (3, 3)), linewidth=.9, color="#9C9690", label="Equal size")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.56, .975), ncol=3,
               frameon=False, fontsize=6.85, handlelength=1.3, handletextpad=.35, columnspacing=.75)
    save(fig, "motivation_3_kv_storage")


def main():
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False}):
        coverage_figure()
        objectives_figure()
        storage_figure()
    print("Rendered three figures: PDF, SVG and opaque 600-dpi PNG.")


if __name__ == "__main__":
    main()
