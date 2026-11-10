"""Plot only audited JSON results for Contextual Cartridges.

The figure deliberately separates three claims:

1. the new query-specific synthetic data reduces dilution;
2. it exposes more of the evidence needed for cross-fact questions;
3. an older evidence-only integration did not improve downstream accuracy,
   while the current TracIn experiment remains pending.

No missing accuracy is imputed and a partial run is never labelled as a
performance result.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


BLUE = {
    "ink": "#0B2A3A",
    "deep": "#1E4B6B",
    "main": "#3180A8",
    "cyan": "#6EC1D8",
    "pale": "#B7E3F1",
}
ORANGE = {
    "yellow": "#F7C948",
    "gold": "#F59E2C",
    "orange": "#E36A1C",
    "rust": "#D35B1D",
    "brown": "#8C3A1A",
}
GRAY = "#667781"
LIGHT_GRID = "#DCE7EC"
ORIGINAL = ORANGE["orange"]
GUIDED = BLUE["main"]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def public_current_status(status: dict[str, Any]) -> dict[str, Any]:
    """Drop machine-local paths while preserving auditable counts and hashes."""
    clean = dict(status)
    clean.pop("root", None)
    report = clean.get("accuracy_report")
    if report:
        clean["accuracy_report"] = Path(report).name
    return clean


def public_export_record(record: dict[str, Any]) -> dict[str, Any]:
    """Make export records portable before publishing them."""
    clean = json.loads(json.dumps(record))
    for output in clean.get("outputs", []):
        if output.get("path"):
            output["path"] = Path(output["path"]).name
    return clean


def validate_quality(report: dict[str, Any]) -> None:
    if report.get("schema") != "query-anchored-dataset-audit-v1":
        raise ValueError("Expected query-anchored-dataset-audit-v1")
    if report.get("completed_cases") != 1000 or report.get("planned_cases") != 1000:
        raise ValueError("Dataset-quality figure requires all 1,000 paired cases")
    if report.get("dataset_gate_passed") is not True:
        raise ValueError("Dataset quality gate did not pass")
    if report.get("benchmark_answers_read") is not False:
        raise ValueError("Dataset audit unexpectedly used benchmark answers")
    for arm in ("vanilla", "new_guided"):
        if arm not in report.get("summaries", {}):
            raise ValueError(f"Missing dataset arm: {arm}")
    cross = report.get("cross_fact_subset", {})
    if cross.get("cases", 0) < 1:
        raise ValueError("Cross-fact subset is empty")
    for arm in ("vanilla", "new_guided"):
        if arm not in cross.get("summaries", {}):
            raise ValueError(f"Missing cross-fact arm: {arm}")


def validate_legacy(report: dict[str, Any]) -> None:
    if report.get("protocol") != "official-Cartridges-selective-objective-aligned-v18":
        raise ValueError("Unexpected evidence-only development-gate protocol")
    overall = report.get("overall", {})
    if overall.get("complete_case_count", 0) < 1:
        raise ValueError("No complete paired evidence-only cases")
    for arm in ("vanilla_syns_vanilla_sd", "specific_syns_vanilla_sd"):
        if arm not in overall.get("arms", {}):
            raise ValueError(f"Missing legacy arm: {arm}")


def percent(value: float) -> float:
    return 100.0 * float(value)


def metric_pair(summary: dict[str, Any], key: str) -> tuple[float, float]:
    return percent(summary["vanilla"][key]), percent(summary["new_guided"][key])


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=BLUE["ink"])
    ax.yaxis.label.set_color(BLUE["ink"])


def grouped_bars(
    ax: plt.Axes,
    labels: list[str],
    original: list[float],
    guided: list[float],
    *,
    title: str,
    subtitle: str,
    delta_directions: list[str],
    ylim: tuple[float, float] = (0, 108),
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    bars_a = ax.bar(
        x - width / 2,
        original,
        width,
        color=ORIGINAL,
        edgecolor=ORANGE["brown"],
        linewidth=0.7,
        hatch="//",
        label="Original Cartridges",
    )
    bars_b = ax.bar(
        x + width / 2,
        guided,
        width,
        color=GUIDED,
        edgecolor=BLUE["ink"],
        linewidth=0.7,
        hatch="..",
        label="Query-specific v3",
    )
    ax.set_ylim(*ylim)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of synthetic questions (%)")
    ax.set_title(title, loc="left", color=BLUE["ink"], fontweight="bold", pad=17)
    ax.text(
        0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=GRAY,
    )
    for bars in (bars_a, bars_b):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                color=BLUE["ink"],
                fontweight="bold",
            )
    for idx, direction in enumerate(delta_directions):
        raw_delta = guided[idx] - original[idx]
        text_delta = raw_delta if direction == "up" else -raw_delta
        glyph = "gain" if direction == "up" else "reduction"
        ax.text(
            idx,
            103.0,
            f"{glyph} {text_delta:+.1f} pp" if direction == "up" else f"{glyph} {text_delta:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=BLUE["deep"],
            fontweight="bold",
        )
    clean_axis(ax)


def extract_current_status(root: Path) -> dict[str, Any]:
    config_path = root / "config.json"
    result_dir = root / "results" / "cases"
    report_path = root / "report" / "accuracy.json"
    planned = None
    protocol = None
    config_sha = None
    if config_path.exists():
        config = read_json(config_path)
        planned = len(config.get("cases", []))
        protocol = config.get("protocol")
        config_sha = digest(config_path)
    completed_files: list[Path] = []
    if result_dir.exists():
        for path in result_dir.glob("*.json"):
            try:
                if read_json(path).get("status") == "completed":
                    completed_files.append(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    status: dict[str, Any] = {
        "root": str(root.resolve()),
        "protocol": protocol,
        "planned": planned,
        "completed_raw_cases": len(completed_files),
        "config_sha256": config_sha,
        "accuracy_available": False,
        "accuracy_report": None,
    }
    if report_path.exists():
        report = read_json(report_path)
        overall = report.get("overall", {})
        arms = overall.get("arms", {})
        wanted = (
            "vanilla_syns_vanilla_sd",
            "specific_syns_vanilla_sd",
            "specific_syns_selective_sd",
        )
        if all(name in arms for name in wanted) and overall.get("complete_case_count", 0) > 0:
            status["accuracy_available"] = True
            status["accuracy_report"] = str(report_path.resolve())
            status["accuracy_sha256"] = digest(report_path)
            status["complete_case_count"] = overall["complete_case_count"]
            status["accuracies"] = {
                name: arms[name]["paired_accuracy_percent"] for name in wanted
            }
    return status


def downstream_panel(
    ax: plt.Axes,
    legacy: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    overall = legacy["overall"]
    old_n = overall["complete_case_count"]
    original_acc = overall["arms"]["vanilla_syns_vanilla_sd"]["paired_accuracy_percent"]
    evidence_acc = overall["arms"]["specific_syns_vanilla_sd"]["paired_accuracy_percent"]
    contrast = overall["contrasts"]["query_specific_with_vanilla_SD"]

    x = np.arange(3)
    bars = ax.bar(
        x[:2],
        [original_acc, evidence_acc],
        width=0.58,
        color=[ORIGINAL, ORANGE["gold"]],
        edgecolor=[ORANGE["brown"], ORANGE["brown"]],
        hatch=["//", "xx"],
        linewidth=0.8,
    )
    for bar, value in zip(bars, (original_acc, evidence_acc)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.1,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.1,
            color=BLUE["ink"],
            fontweight="bold",
        )

    current_has_accuracy = bool(current.get("accuracy_available"))
    if current_has_accuracy:
        current_n = int(current["complete_case_count"])
        ours = float(current["accuracies"]["specific_syns_selective_sd"])
        partial = current.get("planned") is not None and current_n < int(current["planned"])
        ours_bar = ax.bar(
            [2], [ours], width=0.58, color=GUIDED, edgecolor=BLUE["ink"],
            hatch=".." if not partial else "...", linewidth=0.9,
        )[0]
        ax.text(
            ours_bar.get_x() + ours_bar.get_width() / 2,
            ours + 1.1,
            f"{ours:.2f}" + ("\npartial" if partial else ""),
            ha="center", va="bottom", fontsize=8.1, color=BLUE["ink"],
            fontweight="bold",
        )
        current_label = f"Current Contextual\n(n={current_n})"
    else:
        # A dashed outline explicitly encodes missing—not zero—accuracy.
        pending_height = 39.0
        rect = Rectangle(
            (2 - 0.29, 0), 0.58, pending_height,
            facecolor=BLUE["pale"], edgecolor=BLUE["main"],
            linewidth=1.2, linestyle=(0, (4, 3)), hatch="..", alpha=0.32,
        )
        ax.add_patch(rect)
        complete = current.get("completed_raw_cases", 0)
        planned = current.get("planned") or "?"
        ax.text(
            2, pending_height / 2,
            f"PENDING\n{complete}/{planned}\nno estimate",
            ha="center", va="center", fontsize=8.0,
            color=BLUE["deep"], fontweight="bold",
        )
        current_label = "Current Contextual\n(TracIn SD)"

    ax.set_xticks(
        x,
        ["Original\nCartridge", "Direct evidence\n+ Vanilla SD", current_label],
    )
    ax.set_ylim(0, 46)
    ax.set_ylabel("BABILong accuracy (%)")
    ax.set_title(
        "C  Evidence alone failed downstream",
        loc="left", color=BLUE["ink"], fontweight="bold", pad=17,
    )
    ax.text(
        0, 1.015,
        f"Old v18 development gate: n={old_n}/1000 paired cases (partial)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.1, color=GRAY,
    )
    ax.annotate(
        f"paired Δ {contrast['delta_pp']:+.2f} pp\n"
        f"{contrast['fixes']} fixed / {contrast['regressions']} regressed",
        xy=(1, evidence_acc), xytext=(1.05, 42.8), textcoords="data",
        ha="center", va="top", fontsize=7.0, color=ORANGE["brown"],
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": ORANGE["rust"]},
    )
    clean_axis(ax)
    return {
        "n": old_n,
        "original_accuracy_percent": original_acc,
        "direct_evidence_vanilla_sd_accuracy_percent": evidence_acc,
        "paired_delta_pp": contrast["delta_pp"],
        "fixes": contrast["fixes"],
        "regressions": contrast["regressions"],
        "scope_warning": "old v18 partial development gate; not the new v3 data or current TracIn SD",
    }


def build_metrics(
    quality: dict[str, Any],
    legacy: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    summaries = quality["summaries"]
    cross = quality["cross_fact_subset"]
    cross_summaries = cross["summaries"]
    overall_keys = {
        "query_focus_percent": "query_term_hit_fraction_mean",
        "support_recall_lower_bound_percent": "support_recall_lower_bound_mean",
        "all_support_exposed_lower_bound_percent": "all_support_context_fraction_lower_bound_mean",
        "no_support_context_percent": "no_support_context_fraction_mean",
    }
    result: dict[str, Any] = {
        "dataset_audit": {
            "cases": quality["completed_cases"],
            "paired_questions": summaries["vanilla"]["questions"],
            "metrics": {},
        },
        "cross_fact_audit": {
            "cases": cross["cases"],
            "paired_questions": cross_summaries["vanilla"]["questions"],
            "metrics": {},
        },
        "current_tracin": current,
        "claims": {
            "dataset_quality_established": True,
            "direct_evidence_improves_accuracy": False,
            "current_tracin_accuracy_improvement_established": bool(
                current.get("accuracy_available")
                and current.get("complete_case_count") == current.get("planned")
            ),
        },
    }
    for output_name, key in overall_keys.items():
        original, guided = metric_pair(summaries, key)
        result["dataset_audit"]["metrics"][output_name] = {
            "original": original,
            "query_specific_v3": guided,
            "raw_delta_pp": guided - original,
        }
    for output_name, key in overall_keys.items():
        original, guided = metric_pair(cross_summaries, key)
        result["cross_fact_audit"]["metrics"][output_name] = {
            "original": original,
            "query_specific_v3": guided,
            "raw_delta_pp": guided - original,
        }
    # The downstream fields are filled by the plotting function so that its
    # exact plotted values are persisted beside the figure.
    result["legacy_downstream"] = {}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--tracin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output base path without suffix")
    parser.add_argument(
        "--public-manifest",
        action="store_true",
        help="Publish portable manifests without machine-local absolute paths",
    )
    parser.add_argument(
        "--skill-root",
        type=Path,
        default=None,
        help="Scientific-visualization skill root (defaults below the current user's home)",
    )
    args = parser.parse_args()
    if args.skill_root is None:
        args.skill_root = Path.home() / ".codex" / "skills" / "scientific-visualization"

    quality = read_json(args.quality)
    legacy = read_json(args.legacy)
    validate_quality(quality)
    validate_legacy(legacy)
    current = extract_current_status(args.tracin_root)
    metrics = build_metrics(quality, legacy, current)
    if args.public_manifest:
        metrics["current_tracin"] = public_current_status(current)

    sys.path.insert(0, str(args.skill_root / "scripts"))
    from figure_export import export_figure
    from style_presets import style_context

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with style_context("default"):
        mpl.rcParams.update({
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.4,
        })
        fig, axes = plt.subplots(
            1, 3, figsize=(13.3, 4.35),
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.12]},
        )
        fig.subplots_adjust(left=0.055, right=0.985, top=0.71, bottom=0.235, wspace=0.31)

        overall = quality["summaries"]
        query_original, query_guided = metric_pair(overall, "query_term_hit_fraction_mean")
        no_original, no_guided = metric_pair(overall, "no_support_context_fraction_mean")
        grouped_bars(
            axes[0],
            ["Questions containing\na query term ↑", "Contexts containing\nno support fact ↓"],
            [query_original, no_original],
            [query_guided, no_guided],
            title="A  Less dilution",
            subtitle="1,000 cases · 26,451 count-matched questions",
            delta_directions=["up", "down"],
        )

        cross = quality["cross_fact_subset"]["summaries"]
        recall_original, recall_guided = metric_pair(cross, "support_recall_lower_bound_mean")
        together_original, together_guided = metric_pair(
            cross, "all_support_context_fraction_lower_bound_mean"
        )
        grouped_bars(
            axes[1],
            ["Supporting facts\ncovered ↑", "Questions showing all\nsupport facts ↑"],
            [recall_original, together_original],
            [recall_guided, together_guided],
            title="B  Less fragmentation",
            subtitle="352 cross-fact cases · 10,090 paired questions",
            delta_directions=["up", "up"],
        )

        metrics["legacy_downstream"] = downstream_panel(axes[2], legacy, current)

        legend = [
            Patch(facecolor=ORIGINAL, edgecolor=ORANGE["brown"], hatch="//", label="Original Cartridges"),
            Patch(facecolor=GUIDED, edgecolor=BLUE["ink"], hatch="..", label="Query-specific v3"),
        ]
        fig.legend(
            handles=legend, loc="upper left", bbox_to_anchor=(0.055, 0.825),
            ncol=2, frameon=False, handlelength=1.8, columnspacing=1.5,
        )
        fig.suptitle(
            "Direct measurements from the saved JSON",
            x=0.055, y=0.965, ha="left", fontsize=14.0,
            color=BLUE["ink"], fontweight="bold",
        )
        fig.text(
            0.055, 0.905,
            "Query-specific data are substantially more focused and evidence-complete; "
            "direct evidence use alone did not improve the older downstream gate.",
            ha="left", va="center", fontsize=8.8, color=GRAY,
        )
        fig.text(
            0.055, 0.055,
            "A–B: percentages computed directly from saved questions and BABILong support annotations; no answer labels used.  "
            "C: older v18 partial development gate; it used the previous guided data, not v3.  "
            f"Current v26 TracIn result: {current.get('completed_raw_cases', 0)}/{current.get('planned') or '?'} raw cases; "
            "no performance claim until the paired report completes.",
            ha="left", va="bottom", fontsize=6.8, color=GRAY,
        )

        provenance_current = public_current_status(current) if args.public_manifest else current
        provenance = {
            "quality_report": args.quality.name if args.public_manifest else str(args.quality.resolve()),
            "quality_sha256": digest(args.quality),
            "legacy_report": args.legacy.name if args.public_manifest else str(args.legacy.resolve()),
            "legacy_sha256": digest(args.legacy),
            "current_tracin": provenance_current,
            "script": Path(__file__).name if args.public_manifest else str(Path(__file__).resolve()),
            "script_sha256": digest(Path(__file__)),
            "palette": {"blue": BLUE, "orange": ORANGE},
            "missing_data": "Current TracIn accuracy is shown as pending; no imputation.",
            "statistics": "Only observed percentages, percentage-point differences, and paired fix/regression counts are displayed.",
        }
        export = export_figure(
            fig,
            args.output,
            formats=["pdf", "svg", "png"],
            dpi=400,
            bbox_inches=None,
            font_mode="truetype",
            overwrite=True,
            provenance=provenance,
            write_manifest=True,
        )
        plt.close(fig)

    public_export = public_export_record(export) if args.public_manifest else export
    metrics["sources"] = {
        "quality": {
            "path": args.quality.name if args.public_manifest else str(args.quality.resolve()),
            "sha256": digest(args.quality),
        },
        "legacy": {
            "path": args.legacy.name if args.public_manifest else str(args.legacy.resolve()),
            "sha256": digest(args.legacy),
        },
    }
    metrics["figure"] = {
        "base": args.output.name if args.public_manifest else str(args.output.resolve()),
        "exports": public_export,
        "alt_text": (
            "Three-panel BABILong result figure. Query-specific v3 synthetic data greatly increase "
            "query focus and evidence exposure relative to Original Cartridges. An older partial "
            "development gate shows direct evidence plus Vanilla SD below Original Cartridges, while "
            "the current TracIn-selective result is explicitly marked pending."
        ),
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.public_manifest:
        export_path = args.output.with_suffix(".export.json")
        export_report = read_json(export_path)
        export_report = public_export_record(export_report)
        export_report["provenance"] = provenance
        export_path.write_text(
            json.dumps(export_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "figure_base": str(args.output),
        "metrics": str(metrics_path),
        "current_completed": current.get("completed_raw_cases"),
        "current_planned": current.get("planned"),
    }))


if __name__ == "__main__":
    main()
