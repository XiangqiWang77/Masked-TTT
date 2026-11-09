#!/usr/bin/env python3
"""Create the three publication figures for Answer-Guided v3.

The script deliberately separates a mechanism schematic from measured data.
Figures 2 and 3 are generated only from completed per-case records. Missing
values remain missing and are never converted to zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


SKILL_SCRIPTS = Path(
    "/home/da839/.codex/skills/scientific-visualization/scripts"
)
if SKILL_SCRIPTS.exists():
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from figure_export import export_figure
    from style_presets import style_context
else:  # Portable fallback for users without the Codex skill package.
    from contextlib import contextmanager

    @contextmanager
    def style_context(*_args: Any, **_kwargs: Any):
        with plt.rc_context(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "font.family": "sans-serif",
                "font.size": 8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        ):
            yield {"style": "portable-fallback"}

    def export_figure(
        fig: Any,
        filename: str | Path,
        *,
        formats: Iterable[str],
        dpi: float,
        overwrite: bool,
        mkdir: bool,
        write_manifest: bool,
        provenance: dict[str, Any],
        metadata: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        base = Path(filename)
        if mkdir:
            base.parent.mkdir(parents=True, exist_ok=True)
        outputs = []
        for suffix in formats:
            path = base.with_suffix(f".{suffix}")
            if path.exists() and not overwrite:
                raise FileExistsError(path)
            fig.savefig(path, dpi=dpi, facecolor="white")
            outputs.append({"path": str(path), "format": suffix})
        manifest = {
            "outputs": outputs,
            "provenance": provenance,
            "metadata": metadata,
            "fallback_export": True,
        }
        if write_manifest:
            base.with_suffix(".export.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return manifest


VANILLA = "Question-agnostic"
GUIDED = "Query-specific"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#8A8A8A"
LIGHT_GREY = "#E8E8E8"
DARK = "#222222"
LENGTH_ORDER = [8192, 16384, 32768, 65536]
LENGTH_LABELS = {8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--vanilla-timing-dir", type=Path, required=True)
    parser.add_argument("--guided-timing-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_svg_and_export_record(base: Path) -> None:
    """Remove generator-only line-end spaces and refresh recorded byte size."""
    svg_path = base.with_suffix(".svg")
    normalized = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    ) + "\n"
    svg_path.write_text(normalized, encoding="utf-8")
    record_path = base.with_suffix(".export.json")
    record = read_json(record_path)
    for output in record.get("outputs", []):
        if output.get("format") == "svg":
            output["size_bytes"] = svg_path.stat().st_size
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def mean_valid(values: Iterable[float | None]) -> tuple[float, int]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return math.nan, 0
    return statistics.fmean(valid), len(valid)


def median_valid(values: Iterable[float | None]) -> tuple[float, int]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return math.nan, 0
    return statistics.median(valid), len(valid)


def load_timing_rows(root: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        record = read_json(path)
        seconds = record.get("seconds")
        timing_note = "case wall time"
        if seconds is None:
            timing = record.get("timings", {})
            seconds = timing.get("case_seconds")
            if timing.get("wall_time_includes_waiting_for_shared_concurrent_batches"):
                timing_note = "case wall time; includes shared-batch waiting"
        if seconds is None:
            continue
        length = int(record["length"])
        rows.append(
            {
                "case": record.get("case", path.stem),
                "length_tokens": length,
                "length_label": LENGTH_LABELS.get(length, str(length)),
                "method": method,
                "seconds": float(seconds),
                "worker_rank": record.get("worker_rank", ""),
                "timing_note": timing_note,
            }
        )
    return rows


def load_quality_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        for source_key, method in (("vanilla", VANILLA), ("new_guided", GUIDED)):
            local = case[source_key]["local"]
            support = case[source_key]["support"]
            rows.append(
                {
                    "case": case["case"],
                    "task": case["task"],
                    "length_tokens": int(case["length"]),
                    "length_label": LENGTH_LABELS[int(case["length"])],
                    "method": method,
                    "query_focus_fraction": local.get("query_term_hit_fraction"),
                    "support_recall_lower_bound": support.get(
                        "context_union_support_recall_min"
                    ),
                    "complete_chain_exposure_lower_bound": support.get(
                        "complete_context_fraction_min"
                    ),
                    "no_support_context_fraction": support.get(
                        "no_support_context_fraction"
                    ),
                    "mean_evidence_characters": support.get("mean_evidence_characters"),
                    "mean_question_words": support.get("mean_question_words"),
                    "multi_chunk_evidence_fraction": local.get(
                        "multi_chunk_evidence_fraction"
                    ),
                }
            )
    return rows


def aggregate_timing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in (VANILLA, GUIDED):
        method_rows = [row for row in rows if row["method"] == method]
        for length in [None, *LENGTH_ORDER]:
            subset = (
                method_rows
                if length is None
                else [row for row in method_rows if row["length_tokens"] == length]
            )
            values = [row["seconds"] for row in subset]
            quartiles = statistics.quantiles(values, n=4, method="inclusive")
            output.append(
                {
                    "method": method,
                    "length_tokens": "all" if length is None else length,
                    "length_label": "All" if length is None else LENGTH_LABELS[length],
                    "n": len(values),
                    "mean_seconds": statistics.fmean(values),
                    "median_seconds": statistics.median(values),
                    "q1_seconds": quartiles[0],
                    "q3_seconds": quartiles[2],
                }
            )
    return output


def aggregate_quality(
    rows: list[dict[str, Any]], report: dict[str, Any]
) -> list[dict[str, Any]]:
    metrics = [
        "query_focus_fraction",
        "support_recall_lower_bound",
        "complete_chain_exposure_lower_bound",
        "no_support_context_fraction",
        "mean_evidence_characters",
        "mean_question_words",
    ]
    output: list[dict[str, Any]] = []
    summary_map = {
        VANILLA: report["summaries"]["vanilla"],
        GUIDED: report["summaries"]["new_guided"],
    }
    summary_metric_map = {
        "query_focus_fraction": "query_term_hit_fraction_mean",
        "support_recall_lower_bound": "support_recall_lower_bound_mean",
        "complete_chain_exposure_lower_bound": (
            "all_support_context_fraction_lower_bound_mean"
        ),
        "no_support_context_fraction": "no_support_context_fraction_mean",
    }
    for method in (VANILLA, GUIDED):
        method_rows = [row for row in rows if row["method"] == method]
        for length in [None, *LENGTH_ORDER]:
            subset = (
                method_rows
                if length is None
                else [row for row in method_rows if row["length_tokens"] == length]
            )
            for metric in metrics:
                value, n = mean_valid(row.get(metric) for row in subset)
                if length is None and metric in summary_metric_map:
                    # Use the released audit's exact published aggregate.
                    value = float(summary_map[method][summary_metric_map[metric]])
                output.append(
                    {
                        "method": method,
                        "length_tokens": "all" if length is None else length,
                        "length_label": "All" if length is None else LENGTH_LABELS[length],
                        "metric": metric,
                        "mean": value,
                        "valid_n": n,
                        "planned_n": len(subset),
                    }
                )
    return output


def rounded_box(
    ax: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    text: str,
    text_color: str = DARK,
    fontsize: float = 7.5,
    linewidth: float = 1.0,
    zorder: int = 2,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        weight="bold",
        zorder=zorder + 1,
    )


def arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GREY,
    width: float = 1.0,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=width,
            color=color,
            connectionstyle=connectionstyle,
            zorder=1,
        )
    )


def draw_document(ax: Any, x: float, y: float, width: float, height: float) -> None:
    chunk_gap = 0.006
    chunks = 12
    chunk_w = (width - chunk_gap * (chunks - 1)) / chunks
    evidence_colors = {2: BLUE, 8: VERMILLION, 9: VERMILLION}
    for index in range(chunks):
        color = evidence_colors.get(index, "#D9D9D9")
        ax.add_patch(
            Rectangle(
                (x + index * (chunk_w + chunk_gap), y),
                chunk_w,
                height,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
        )
    ax.text(x, y + height + 0.018, "Long document $D$", fontsize=7, weight="bold")


def draw_memory(ax: Any, x: float, y: float, focused: bool) -> None:
    cols, rows = 6, 2
    for row in range(rows):
        for col in range(cols):
            if focused:
                colors = [BLUE, VERMILLION, GREEN]
                color = colors[(row * cols + col) % len(colors)]
                alpha = 0.82
            else:
                color = ["#BDBDBD", "#D7D7D7", BLUE, "#C8C8C8", VERMILLION][
                    (row * cols + col) % 5
                ]
                alpha = 0.38 if color in (BLUE, VERMILLION) else 0.75
            ax.add_patch(
                FancyBboxPatch(
                    (x + col * 0.023, y + row * 0.044),
                    0.018,
                    0.034,
                    boxstyle="round,pad=0.002",
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.4,
                    alpha=alpha,
                )
            )
    ax.text(x + 0.058, y - 0.025, "fixed-size KV", ha="center", fontsize=6.5)


def plot_mechanism(output_dir: Path, provenance: dict[str, Any], overwrite: bool) -> None:
    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, ax = plt.subplots(figsize=(7.2, 3.55))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(
            0.01,
            0.965,
            "A  From query-agnostic compression to answer-guided self-study",
            fontsize=10,
            weight="bold",
            va="top",
        )
        ax.text(0.01, 0.82, "QUERY-AGNOSTIC", color=GREY, fontsize=7.5, weight="bold")
        ax.text(0.01, 0.39, "CONTEXTUAL CARTRIDGES", color=BLUE, fontsize=7.5, weight="bold")

        draw_document(ax, 0.04, 0.675, 0.22, 0.065)
        arrow(ax, (0.275, 0.707), (0.335, 0.707))
        rounded_box(
            ax,
            (0.34, 0.64),
            0.145,
            0.135,
            facecolor="#F3F3F3",
            edgecolor="#AAAAAA",
            text="block-local\nquestions",
            text_color="#666666",
        )
        for index in range(8):
            x = 0.515 + (index % 4) * 0.035
            y = 0.675 + (index // 4) * 0.055
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    0.012,
                    facecolor=["#BDBDBD", "#D7D7D7", BLUE, "#C8C8C8"][index % 4],
                    edgecolor="white",
                    linewidth=0.4,
                )
            )
        ax.text(0.565, 0.63, "budget spread across topics", ha="center", fontsize=6.5)
        arrow(ax, (0.655, 0.707), (0.735, 0.707))
        draw_memory(ax, 0.75, 0.665, focused=False)
        rounded_box(
            ax,
            (0.895, 0.66),
            0.09,
            0.10,
            facecolor="#F7E7E2",
            edgecolor=VERMILLION,
            text="Dilution",
            text_color=VERMILLION,
            fontsize=7,
        )
        ax.text(
            0.41,
            0.565,
            "Evidence remains separated → Fragmentation",
            color=VERMILLION,
            fontsize=7,
            ha="center",
        )
        ax.plot([0.27, 0.72], [0.52, 0.52], color="#D0D0D0", linewidth=0.8)

        rounded_box(
            ax,
            (0.04, 0.225),
            0.07,
            0.08,
            facecolor="#E5F1F8",
            edgecolor=BLUE,
            text="query $q$",
            text_color=BLUE,
        )
        draw_document(ax, 0.04, 0.105, 0.22, 0.055)
        arrow(ax, (0.115, 0.265), (0.305, 0.265), color=BLUE)
        arrow(ax, (0.265, 0.135), (0.305, 0.225), color=GREY)
        rounded_box(
            ax,
            (0.31, 0.19),
            0.13,
            0.14,
            facecolor="#E5F1F8",
            edgecolor=BLUE,
            text="Decision model\nselects $E$",
            text_color=BLUE,
        )
        rounded_box(
            ax,
            (0.475, 0.23),
            0.11,
            0.085,
            facecolor="#E5F1F8",
            edgecolor=BLUE,
            text="anchor block",
            text_color=BLUE,
            fontsize=6.7,
        )
        rounded_box(
            ax,
            (0.475, 0.105),
            0.11,
            0.085,
            facecolor="#FCE9DF",
            edgecolor=VERMILLION,
            text="bridge block",
            text_color=VERMILLION,
            fontsize=6.7,
        )
        arrow(ax, (0.44, 0.26), (0.47, 0.27), color=BLUE)
        arrow(ax, (0.44, 0.24), (0.47, 0.15), color=VERMILLION)
        rounded_box(
            ax,
            (0.625, 0.18),
            0.105,
            0.12,
            facecolor="#E2F3EC",
            edgecolor=GREEN,
            text="useful\nconclusion $a$",
            text_color=GREEN,
            fontsize=6.4,
        )
        arrow(ax, (0.585, 0.272), (0.62, 0.257), color=BLUE)
        arrow(ax, (0.585, 0.148), (0.62, 0.215), color=VERMILLION)
        rounded_box(
            ax,
            (0.745, 0.18),
            0.10,
            0.12,
            facecolor="#F3E8F2",
            edgecolor=PURPLE,
            text="reverse\nquestion $s$",
            text_color=PURPLE,
            fontsize=6.4,
        )
        arrow(ax, (0.73, 0.24), (0.74, 0.24), color=GREEN)
        draw_memory(ax, 0.86, 0.19, focused=True)
        arrow(ax, (0.845, 0.24), (0.855, 0.24), color=PURPLE)
        ax.text(0.535, 0.055, "cross-block relation exposed", ha="center", fontsize=6.8, color=GREEN)
        ax.text(0.93, 0.055, "capacity focused on $q$", ha="center", fontsize=6.8, color=BLUE)

        export_figure(
            fig,
            output_dir / "figure1_fragmentation_dilution_mechanism",
            formats=("png", "svg", "pdf"),
            dpi=400,
            overwrite=overwrite,
            mkdir=True,
            bbox_inches=None,
            write_manifest=True,
            metadata={"Title": "Answer-guided mechanism for fragmentation and dilution"},
            provenance={**provenance, "figure_kind": "schematic; no measured values"},
        )
        plt.close(fig)


def lookup_summary(
    rows: list[dict[str, Any]], method: str, length: int | str
) -> dict[str, Any]:
    for row in rows:
        if row["method"] == method and row["length_tokens"] == length:
            return row
    raise KeyError((method, length))


def plot_speed(
    output_dir: Path,
    timing_rows: list[dict[str, Any]],
    timing_summary: list[dict[str, Any]],
    provenance: dict[str, Any],
    overwrite: bool,
) -> None:
    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={"wspace": 0.34})
        fig.suptitle(
            "B  Observed generation wall time (completed 1,000-case campaigns)",
            x=0.02,
            y=0.99,
            ha="left",
            fontsize=10,
            weight="bold",
        )
        ax = axes[0]
        x = np.arange(len(LENGTH_ORDER))
        width = 0.34
        vanilla_medians = [
            lookup_summary(timing_summary, VANILLA, length)["median_seconds"]
            for length in LENGTH_ORDER
        ]
        guided_medians = [
            lookup_summary(timing_summary, GUIDED, length)["median_seconds"]
            for length in LENGTH_ORDER
        ]
        bars_v = ax.bar(
            x - width / 2,
            vanilla_medians,
            width,
            color=GREY,
            edgecolor=DARK,
            linewidth=0.5,
            hatch="//",
            label=VANILLA,
        )
        bars_g = ax.bar(
            x + width / 2,
            guided_medians,
            width,
            color=BLUE,
            edgecolor=DARK,
            linewidth=0.5,
            label=GUIDED,
        )
        ax.bar_label(bars_v, fmt="%.0f", fontsize=5.7, padding=1)
        ax.bar_label(bars_g, fmt="%.0f", fontsize=5.7, padding=1)
        ax.set_xticks(x, [LENGTH_LABELS[length] for length in LENGTH_ORDER])
        ax.set_ylim(0, max(vanilla_medians + guided_medians) * 1.2)
        ax.set_ylabel("Median seconds / case")
        ax.set_title("a  By document length", loc="left", weight="bold")
        ax.legend(loc="upper left", fontsize=5.8)

        ax = axes[1]
        for method, color, linestyle in (
            (VANILLA, GREY, "--"),
            (GUIDED, BLUE, "-"),
        ):
            values = sorted(
                row["seconds"] for row in timing_rows if row["method"] == method
            )
            y = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, y, where="post", color=color, linestyle=linestyle, label=method)
            median = statistics.median(values)
            ax.plot(median, 0.5, marker="D", color=color, markersize=4)
        ax.set_xscale("log")
        ax.set_xlim(left=max(1, min(row["seconds"] for row in timing_rows) * 0.8))
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("Observed case wall time (s, log scale)")
        ax.set_ylabel("Share completed")
        ax.set_title("b  Full distribution", loc="left", weight="bold")
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.5)

        ax = axes[2]
        ratios = [v / g for v, g in zip(vanilla_medians, guided_medians)]
        colors = [GREEN if ratio >= 1 else VERMILLION for ratio in ratios]
        bars = ax.bar(x, ratios, color=colors, edgecolor=DARK, linewidth=0.5)
        ax.axhline(1.0, color=DARK, linewidth=0.8, linestyle="--")
        ax.bar_label(bars, labels=[f"{value:.2f}×" for value in ratios], fontsize=6, padding=2)
        ax.set_xticks(x, [LENGTH_LABELS[length] for length in LENGTH_ORDER])
        ax.set_ylim(0, max(ratios) * 1.22)
        ax.set_ylabel("Vanilla / guided median time")
        ax.set_title("c  Descriptive ratio", loc="left", weight="bold")
        overall_v = lookup_summary(timing_summary, VANILLA, "all")["median_seconds"]
        overall_g = lookup_summary(timing_summary, GUIDED, "all")["median_seconds"]
        ax.text(
            0.98,
            0.95,
            f"overall median: {overall_v / overall_g:.2f}×",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
            color=DARK,
        )
        ax.text(
            0.5,
            -0.28,
            "Green: guided faster  •  Orange: guided slower",
            transform=ax.transAxes,
            ha="center",
            fontsize=5.8,
        )
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.82)
        export_figure(
            fig,
            output_dir / "figure2_generation_speed",
            formats=("png", "svg", "pdf"),
            dpi=400,
            overwrite=overwrite,
            mkdir=True,
            bbox_inches=None,
            write_manifest=True,
            metadata={"Title": "Observed synthetic-data generation wall time"},
            provenance={
                **provenance,
                "timing_caveat": (
                    "separate campaigns with different concurrency/backend scheduling; "
                    "query-specific case time includes shared concurrent-batch waiting"
                ),
            },
        )
        plt.close(fig)


def quality_value(
    rows: list[dict[str, Any]], method: str, length: int | str, metric: str
) -> tuple[float, int, int]:
    for row in rows:
        if (
            row["method"] == method
            and row["length_tokens"] == length
            and row["metric"] == metric
        ):
            return float(row["mean"]), int(row["valid_n"]), int(row["planned_n"])
    raise KeyError((method, length, metric))


def plot_quality(
    output_dir: Path,
    quality_summary: list[dict[str, Any]],
    provenance: dict[str, Any],
    overwrite: bool,
) -> None:
    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1))
        fig.suptitle(
            "C  Useful query-specific synthetic data: focus, coverage, and length",
            x=0.025,
            y=0.99,
            ha="left",
            fontsize=10,
            weight="bold",
        )
        methods = (VANILLA, GUIDED)
        styles = {
            VANILLA: {"color": GREY, "linestyle": "--", "marker": "s"},
            GUIDED: {"color": BLUE, "linestyle": "-", "marker": "o"},
        }
        ax = axes[0, 0]
        metrics = [
            ("query_focus_fraction", "Query\nfocus"),
            ("support_recall_lower_bound", "Support\nrecall LB"),
            ("complete_chain_exposure_lower_bound", "Complete-chain\nexposure LB"),
        ]
        x = np.arange(len(metrics))
        width = 0.34
        for offset, method, hatch in ((-width / 2, VANILLA, "//"), (width / 2, GUIDED, "")):
            values = [quality_value(quality_summary, method, "all", metric)[0] * 100 for metric, _ in metrics]
            bars = ax.bar(
                x + offset,
                values,
                width,
                color=styles[method]["color"],
                edgecolor=DARK,
                linewidth=0.5,
                hatch=hatch,
                label=method,
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=5.7, padding=1)
        ax.set_xticks(x, [label for _, label in metrics])
        ax.set_ylim(0, 110)
        ax.set_ylabel("Case-level mean (%)")
        ax.set_title("a  Overall usefulness proxies", loc="left", weight="bold")
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.5)

        def line_panel(axis: Any, metric: str, title: str, ylabel: str, percent: bool = True) -> None:
            for method in methods:
                values = [
                    quality_value(quality_summary, method, length, metric)[0]
                    for length in LENGTH_ORDER
                ]
                if percent:
                    values = [value * 100 for value in values]
                axis.plot(
                    range(len(LENGTH_ORDER)),
                    values,
                    label=method,
                    color=styles[method]["color"],
                    linestyle=styles[method]["linestyle"],
                    marker=styles[method]["marker"],
                    markerfacecolor="white" if method == VANILLA else styles[method]["color"],
                )
            axis.set_xticks(range(len(LENGTH_ORDER)), [LENGTH_LABELS[length] for length in LENGTH_ORDER])
            axis.set_xlabel("Document length")
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", weight="bold")
            axis.grid(axis="y", color="#E4E4E4", linewidth=0.5)
            if percent:
                axis.set_ylim(0, 105)

        line_panel(
            axes[0, 1],
            "support_recall_lower_bound",
            "b  Support recall lower bound",
            "Case-level mean (%)",
        )
        line_panel(
            axes[1, 0],
            "complete_chain_exposure_lower_bound",
            "c  Complete-chain exposure lower bound",
            "Case-level mean (%)",
        )
        line_panel(
            axes[1, 1],
            "mean_evidence_characters",
            "d  Mean evidence footprint",
            "Characters / synthetic item",
            percent=False,
        )
        axes[1, 1].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _position: f"{value / 1000:.0f}K")
        )
        axes[0, 1].text(
            0.02,
            0.05,
            "Vanilla valid n: 249/250, 250/250,\n247/250, 248/250; guided: 250/250",
            transform=axes[0, 1].transAxes,
            fontsize=5.5,
            color="#555555",
        )
        axes[1, 0].text(
            0.98,
            0.05,
            "n=250 cases per point",
            transform=axes[1, 0].transAxes,
            ha="right",
            fontsize=5.5,
            color="#555555",
        )
        axes[1, 1].text(
            0.98,
            0.50,
            "lower is more compact",
            transform=axes[1, 1].transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#555555",
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.51, 0.945),
            ncol=2,
            fontsize=6.3,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.86, hspace=0.43, wspace=0.30)
        export_figure(
            fig,
            output_dir / "figure3_quality_coverage_length",
            formats=("png", "svg", "pdf"),
            dpi=400,
            overwrite=overwrite,
            mkdir=True,
            bbox_inches=None,
            write_manifest=True,
            metadata={"Title": "Synthetic-data quality proxies across document length"},
            provenance={
                **provenance,
                "quality_caveat": (
                    "coverage and focus are audit proxies, not semantic correctness or "
                    "downstream Cartridge accuracy"
                ),
            },
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = read_json(args.quality_report)
    if report.get("completed_cases") != 1000:
        raise ValueError("quality report must contain all 1,000 completed cases")
    vanilla_timing = load_timing_rows(args.vanilla_timing_dir, VANILLA)
    guided_timing = load_timing_rows(args.guided_timing_dir, GUIDED)
    if len(vanilla_timing) != 1000 or len(guided_timing) != 1000:
        raise ValueError(
            f"expected 1,000 timing rows per method, got "
            f"{len(vanilla_timing)} and {len(guided_timing)}"
        )
    timing_rows = vanilla_timing + guided_timing
    quality_rows = load_quality_rows(report)
    timing_summary = aggregate_timing(timing_rows)
    quality_summary = aggregate_quality(quality_rows, report)

    write_csv(
        args.output_dir / "source_data_speed_per_case.csv",
        timing_rows,
        [
            "case",
            "length_tokens",
            "length_label",
            "method",
            "seconds",
            "worker_rank",
            "timing_note",
        ],
    )
    write_csv(
        args.output_dir / "source_data_speed_summary.csv",
        timing_summary,
        [
            "method",
            "length_tokens",
            "length_label",
            "n",
            "mean_seconds",
            "median_seconds",
            "q1_seconds",
            "q3_seconds",
        ],
    )
    write_csv(
        args.output_dir / "source_data_quality_per_case.csv",
        quality_rows,
        [
            "case",
            "task",
            "length_tokens",
            "length_label",
            "method",
            "query_focus_fraction",
            "support_recall_lower_bound",
            "complete_chain_exposure_lower_bound",
            "no_support_context_fraction",
            "mean_evidence_characters",
            "mean_question_words",
            "multi_chunk_evidence_fraction",
        ],
    )
    write_csv(
        args.output_dir / "source_data_quality_summary.csv",
        quality_summary,
        [
            "method",
            "length_tokens",
            "length_label",
            "metric",
            "mean",
            "valid_n",
            "planned_n",
        ],
    )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_report_sha256": sha256_file(args.quality_report),
        "timing_rows_sha256": stable_rows_sha256(timing_rows),
        "quality_rows_sha256": stable_rows_sha256(quality_rows),
        "quality_cases": report["completed_cases"],
        "timing_cases_per_method": 1000,
        "benchmark_answers_read": report.get("benchmark_answers_read"),
        "missing_values_as_zero": False,
    }
    plot_mechanism(args.output_dir, provenance, args.overwrite)
    plot_speed(args.output_dir, timing_rows, timing_summary, provenance, args.overwrite)
    plot_quality(args.output_dir, quality_summary, provenance, args.overwrite)
    for name in (
        "figure1_fragmentation_dilution_mechanism",
        "figure2_generation_speed",
        "figure3_quality_coverage_length",
    ):
        normalize_svg_and_export_record(args.output_dir / name)

    manifest = {
        "schema": "contextual-cartridges-answer-guided-v3-figures-v1",
        "provenance": provenance,
        "figures": {
            "figure1": {
                "kind": "mechanism schematic",
                "base": "figure1_fragmentation_dilution_mechanism",
                "measured_values": False,
            },
            "figure2": {
                "kind": "observed timing",
                "base": "figure2_generation_speed",
                "source": "source_data_speed_per_case.csv",
                "campaigns_controlled": False,
            },
            "figure3": {
                "kind": "quality audit proxies",
                "base": "figure3_quality_coverage_length",
                "source": "source_data_quality_per_case.csv",
                "downstream_accuracy": False,
            },
        },
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overall_v = lookup_summary(timing_summary, VANILLA, "all")["median_seconds"]
    overall_g = lookup_summary(timing_summary, GUIDED, "all")["median_seconds"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "overall_median_seconds": {VANILLA: overall_v, GUIDED: overall_g},
                "descriptive_overall_median_ratio": overall_v / overall_g,
                "quality_report_sha256": provenance["quality_report_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
