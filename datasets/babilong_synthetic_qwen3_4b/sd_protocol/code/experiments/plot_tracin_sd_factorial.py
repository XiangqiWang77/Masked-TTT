"""Render publication-ready figures for the four-arm TracIn SD factorial.

The renderer consumes only the post-inference ``accuracy.json`` report and the
per-case TracIn audit JSON files.  It refuses to label an incomplete run as a
final result unless ``--allow-partial`` is explicitly supplied.  Missing
observations remain missing; they are never converted to zero.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ARM_ORDER = (
    "vanilla_syns_vanilla_sd",
    "vanilla_syns_selective_sd",
    "specific_syns_vanilla_sd",
    "specific_syns_selective_sd",
)
ARM_LABELS = {
    "vanilla_syns_vanilla_sd": "Agnostic + Vanilla SD",
    "vanilla_syns_selective_sd": "Agnostic + Selective SD",
    "specific_syns_vanilla_sd": "Query-specific + Vanilla SD",
    "specific_syns_selective_sd": "Contextual Cartridges",
}
ARM_SHORT = {
    "vanilla_syns_vanilla_sd": "Agnostic\nVanilla",
    "vanilla_syns_selective_sd": "Agnostic\nSelective",
    "specific_syns_vanilla_sd": "Specific\nVanilla",
    "specific_syns_selective_sd": "Specific\nSelective",
}
ARM_COLORS = {
    "vanilla_syns_vanilla_sd": "#777777",
    "vanilla_syns_selective_sd": "#0072B2",
    "specific_syns_vanilla_sd": "#E69F00",
    "specific_syns_selective_sd": "#009E73",
}
ARM_MARKERS = dict(zip(ARM_ORDER, ("o", "s", "^", "D")))
ARM_LINESTYLES = dict(zip(ARM_ORDER, ("-", "--", "-.", ":")))
ARM_HATCHES = dict(zip(ARM_ORDER, ("", "//", "..", "xx")))

CONTRAST_ORDER = (
    "query_specific_with_vanilla_SD",
    "selective_with_vanilla_syns",
    "selective_with_specific_syns",
    "contextual_cartridges_vs_vanilla_cartridges",
)
CONTRAST_LABELS = {
    "query_specific_with_vanilla_SD": "Query-specific data\n(Vanilla SD)",
    "selective_with_vanilla_syns": "Selective SD\n(agnostic data)",
    "selective_with_specific_syns": "Selective SD\n(specific data)",
    "contextual_cartridges_vs_vanilla_cartridges": "Full method vs.\nvanilla Cartridges",
}
DATASET_LABELS = {
    "vanilla_syns": "Question-agnostic",
    "specific_syns": "Query-specific",
}
FIX_COLOR = "#0072B2"
REGRESSION_COLOR = "#D55E00"
KEEP_COLOR = "#009E73"
DELETE_COLOR = "#CC79A7"


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_final_report(report, *, allow_partial=False):
    planned = int(report.get("planned", 0))
    completed = int(report.get("overall", {}).get("complete_case_count", 0))
    complete = bool(report.get("complete")) and planned > 0 and completed == planned
    if not complete and not allow_partial:
        raise ValueError(
            f"refusing final figures for incomplete report: {completed}/{planned}; "
            "use --allow-partial only for a visibly watermarked diagnostic"
        )
    return complete, completed, planned


def arm_names(report):
    observed = tuple(report.get("matrix", ()))
    if set(observed) != set(ARM_ORDER):
        raise ValueError(f"expected the frozen four-arm matrix, found {observed}")
    return ARM_ORDER


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def accuracy_rows(report):
    rows = []
    scopes = [("overall", "Overall", report["overall"])]
    scopes.extend((f"length_{length}", f"{int(length)//1024}K",
                   report["by_length"][str(length)])
                  for length in (8192, 16384, 32768, 65536))
    for scope, label, values in scopes:
        for arm in ARM_ORDER:
            arm_value = values["arms"][arm]
            rows.append({
                "scope": scope,
                "scope_label": label,
                "arm": arm,
                "arm_label": ARM_LABELS[arm],
                "accuracy_percent": finite_or_none(
                    arm_value.get("paired_accuracy_percent")),
                "paired_n": int(arm_value.get("complete_paired_n", 0)),
                "paired_correct": int(arm_value.get("complete_paired_correct", 0)),
            })
    return rows


def contrast_rows(report):
    values = report["overall"]["contrasts"]
    rows = []
    for name in CONTRAST_ORDER:
        value = values.get(name)
        rows.append({
            "contrast": name,
            "contrast_label": CONTRAST_LABELS[name].replace("\n", " "),
            "n": int(value["n"]) if value else 0,
            "fixes": int(value["fixes"]) if value else None,
            "regressions": int(value["regressions"]) if value else None,
            "net_delta_pp": finite_or_none(value.get("delta_pp")) if value else None,
        })
    return rows


def oracle_rows(report):
    values = report["overall"]["oracle_trap_evidence"]
    rows = []
    for dataset in ("vanilla_syns", "specific_syns"):
        value = values.get(dataset)
        cohort = int(value.get("strict_oracle_escape_cohort", 0)) if value else 0
        fixes = int(value.get("strict_oracle_escape_fixes", 0)) if value else None
        regressions = int(value.get("strict_oracle_escape_regressions", 0)) if value else None
        rows.append({
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "strict_cohort_n": cohort,
            "fixes": fixes,
            "regressions": regressions,
            "fix_percent": 100 * fixes / cohort if cohort and fixes is not None else None,
            "regression_percent": (
                100 * regressions / cohort
                if cohort and regressions is not None else None),
            "net_delta_pp": finite_or_none(
                value.get("strict_oracle_escape_delta_pp")) if value else None,
        })
    return rows


def tracin_rows(experiment_root: Path):
    """Load raw candidate decisions; absent audits stay absent, never zero."""
    rows = []
    for path in sorted((experiment_root / "cases").glob("*/tracin/*.json")):
        audit = read_json(path)
        dataset = path.stem
        if dataset not in DATASET_LABELS:
            continue
        decisions = audit.get("decisions", [])
        checkpoint_names = [entry.get("checkpoint", f"checkpoint_{index}")
                            for index, entry in enumerate(audit.get("checkpoints", []))]
        for decision in decisions:
            contributions = decision.get("checkpoint_contributions", [])
            row = {
                "case": path.parents[1].name,
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "candidate_index": int(decision["candidate_index"]),
                "tracin_score": float(decision["tracin_score"]),
                "threshold": float(decision["threshold"]),
                "keep": bool(decision["keep"]),
            }
            for index, contribution in enumerate(contributions):
                name = checkpoint_names[index] if index < len(checkpoint_names) else f"checkpoint_{index}"
                row[f"contribution__{name}"] = float(contribution)
            rows.append(row)
    return rows


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def watermark(fig, complete):
    if not complete:
        fig.text(0.5, 0.5, "PARTIAL — NOT A FINAL RESULT", ha="center",
                 va="center", rotation=24, fontsize=22, color="#A00000",
                 alpha=0.18, fontweight="bold", zorder=100)


def panel_label(ax, label):
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="top")


def mark_missing(ax, x, *, y=2.0):
    ax.scatter([x], [y], marker="x", color="#555555", zorder=5)
    ax.annotate("NA", (x, y), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=6, color="#555555")


def _style_imports():
    skill = Path("/home/da839/.codex/skills/scientific-visualization")
    sys.path[:0] = [str(skill / "scripts"), str(skill / "assets")]
    from figure_export import export_figure
    from style_presets import style_context
    return export_figure, style_context


def _export(export_figure, fig, base: Path, provenance):
    return export_figure(
        fig,
        base,
        formats=["pdf", "svg", "png"],
        dpi=450,
        transparent=False,
        bbox_inches=None,
        facecolor="white",
        font_mode="truetype",
        overwrite=True,
        mkdir=True,
        provenance=provenance,
        write_manifest=True,
    )


def main_results_figure(report, output: Path, *, complete, export_figure,
                        style_context, report_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = accuracy_rows(report)
    lookup = {(row["scope"], row["arm"]): row for row in rows}
    contrasts = contrast_rows(report)
    with style_context("default", palette_name="okabe_ito_on_white"):
        fig = plt.figure(figsize=(178 / 25.4, 126 / 25.4), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.06))
        ax_overall = fig.add_subplot(grid[0, 0])
        ax_length = fig.add_subplot(grid[0, 1])
        ax_changes = fig.add_subplot(grid[1, :])

        x = np.arange(len(ARM_ORDER))
        for index, arm in enumerate(ARM_ORDER):
            row = lookup[("overall", arm)]
            value = row["accuracy_percent"]
            if value is None:
                mark_missing(ax_overall, index)
                continue
            ax_overall.bar(index, value, width=0.72, color=ARM_COLORS[arm],
                           edgecolor="black", hatch=ARM_HATCHES[arm], linewidth=0.6)
            ax_overall.text(index, value + 1.8, f"{value:.1f}", ha="center",
                            va="bottom", fontsize=7, fontweight="bold")
        ax_overall.set(xticks=x, xticklabels=[ARM_SHORT[a] for a in ARM_ORDER],
                       ylim=(0, 100), ylabel="BABILong accuracy (%)",
                       title="Four-arm factorial (same completed cases)")
        ax_overall.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        n = report["overall"]["complete_case_count"]
        ax_overall.text(0.99, 0.03, f"n = {n} paired cases", transform=ax_overall.transAxes,
                        ha="right", va="bottom", fontsize=6.5, color="#444444")
        panel_label(ax_overall, "A")

        lengths = (8192, 16384, 32768, 65536)
        labels = ("8K", "16K", "32K", "64K")
        for arm in ARM_ORDER:
            values = [lookup[(f"length_{length}", arm)]["accuracy_percent"]
                      for length in lengths]
            numeric = [np.nan if value is None else value for value in values]
            ax_length.plot(range(4), numeric, color=ARM_COLORS[arm],
                           marker=ARM_MARKERS[arm], linestyle=ARM_LINESTYLES[arm],
                           label=ARM_LABELS[arm])
            for index, value in enumerate(values):
                if value is None:
                    mark_missing(ax_length, index)
        ax_length.set(xticks=range(4), xticklabels=labels, ylim=(0, 100),
                      xlabel="Document length (tokens)", ylabel="Accuracy (%)",
                      title="Accuracy by document length")
        ax_length.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        legend_handles = [Line2D([0], [0], color=ARM_COLORS[a], marker=ARM_MARKERS[a],
                                 linestyle=ARM_LINESTYLES[a], label=ARM_LABELS[a])
                          for a in ARM_ORDER]
        ax_length.legend(handles=legend_handles, loc="lower left", fontsize=5.8,
                         ncol=1, handlelength=2.7)
        panel_label(ax_length, "B")

        present = [row for row in contrasts if row["fixes"] is not None]
        positions = np.arange(len(CONTRAST_ORDER))
        maximum = 1
        for index, name in enumerate(CONTRAST_ORDER):
            row = next((value for value in present if value["contrast"] == name), None)
            if row is None:
                ax_changes.scatter([0], [index], marker="x", color="#555555")
                continue
            fixes, regressions = row["fixes"], row["regressions"]
            maximum = max(maximum, fixes, regressions)
            ax_changes.barh(index, fixes, color=FIX_COLOR, edgecolor="black",
                            height=0.58, label="Fixes" if index == 0 else None)
            ax_changes.barh(index, -regressions, color=REGRESSION_COLOR,
                            edgecolor="black", hatch="//", height=0.58,
                            label="Regressions" if index == 0 else None)
            net = fixes - regressions
            ax_changes.scatter(net, index, color="black", marker="D", s=24,
                               zorder=5, label="Net" if index == 0 else None)
            ax_changes.text(fixes + 0.35, index, str(fixes), va="center", ha="left", fontsize=6.5)
            ax_changes.text(-regressions - 0.35, index, str(regressions), va="center", ha="right", fontsize=6.5)
        limit = maximum + max(2, math.ceil(maximum * 0.18))
        ax_changes.axvline(0, color="black", linewidth=0.7)
        ax_changes.set(xlim=(-limit, limit), yticks=positions,
                       yticklabels=[CONTRAST_LABELS[name] for name in CONTRAST_ORDER],
                       xlabel="Paired case count (left: regression; right: fix)",
                       title="Paired changes expose both benefit and harm")
        ax_changes.invert_yaxis()
        ax_changes.grid(axis="x", color="#D9D9D9", linewidth=0.5)
        ax_changes.legend(handles=[
            Patch(facecolor=FIX_COLOR, edgecolor="black", label="Fixes"),
            Patch(facecolor=REGRESSION_COLOR, edgecolor="black", hatch="//", label="Regressions"),
            Line2D([0], [0], marker="D", color="black", linestyle="none", label="Net"),
        ], ncol=3, loc="lower right")
        panel_label(ax_changes, "C")
        watermark(fig, complete)

        provenance = {
            "audience": "provisional ACL-style manuscript figure; publisher rules pending verification",
            "raw_data": str(report_path),
            "transformations": [
                "accuracy: exact correct / same completed paired cases",
                "fix: baseline wrong and target correct",
                "regression: baseline correct and target wrong",
            ],
            "uncertainty": "none displayed; fixed development cohort, not repeated random runs",
            "missing_data": "rendered as NA markers; never converted to zero",
            "complete": complete,
        }
        manifest = _export(export_figure, fig, output / "figure_main_results", provenance)
        plt.close(fig)
    return manifest


def selective_diagnostics_figure(report, tracin, output: Path, *, complete,
                                 export_figure, style_context, report_path,
                                 experiment_root):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    oracle = oracle_rows(report)
    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(1, 3, figsize=(178 / 25.4, 70 / 25.4),
                                 layout="constrained")

        ax = axes[0]
        x = np.arange(2)
        width = 0.34
        for index, row in enumerate(oracle):
            if row["fix_percent"] is None:
                mark_missing(ax, index)
                continue
            ax.bar(index - width / 2, row["fix_percent"], width, color=FIX_COLOR,
                   edgecolor="black", label="Fixes" if index == 0 else None)
            ax.bar(index + width / 2, -row["regression_percent"], width,
                   color=REGRESSION_COLOR, edgecolor="black", hatch="//",
                   label="Regressions" if index == 0 else None)
            ax.text(index - width / 2, row["fix_percent"] + 1,
                    str(row["fixes"]), ha="center", va="bottom", fontsize=6.5)
            ax.text(index + width / 2, -row["regression_percent"] - 1,
                    str(row["regressions"]), ha="center", va="top", fontsize=6.5)
            ax.text(index, -47, f"cohort n={row['strict_cohort_n']}", ha="center",
                    va="bottom", fontsize=6, color="#444444")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(xticks=x, xticklabels=[row["dataset_label"] for row in oracle],
               ylim=(-50, 50), ylabel="Strict Oracle-escape cohort (%)",
               title="Teacher-wrong & initial-wrong cases")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=6)
        panel_label(ax, "A")

        ax = axes[1]
        checkpoints = ("initial_cartridge", "vanilla_SD")
        offsets = (-0.18, 0.18)
        for dataset_index, dataset in enumerate(("vanilla_syns", "specific_syns")):
            subset = [row for row in tracin if row["dataset"] == dataset]
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                field = f"contribution__{checkpoint}"
                values = [row[field] for row in subset if field in row
                          and math.isfinite(row[field]) and abs(row[field]) < 1e29]
                position = checkpoint_index + offsets[dataset_index]
                if not values:
                    mark_missing(ax, position)
                    continue
                positive = sum(value > 0 for value in values)
                percent = 100 * positive / len(values)
                ax.bar(position, percent, 0.32, color=("#56B4E9", "#009E73")[dataset_index],
                       edgecolor="black", hatch=("", "//")[dataset_index])
                ax.text(position, percent + 1.5, f"{percent:.0f}%\n(n={len(values)})",
                        ha="center", va="bottom", fontsize=6)
        ax.set(xticks=(0, 1), xticklabels=("Initial KV", "Vanilla-SD KV"),
               ylim=(0, 110), ylabel="Positive candidate influence (%)",
               title="Signed influence across checkpoints")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.legend(handles=[
            Patch(facecolor="#56B4E9", edgecolor="black", label="Question-agnostic"),
            Patch(facecolor="#009E73", edgecolor="black", hatch="//", label="Query-specific"),
        ], loc="lower right", fontsize=5.8)
        panel_label(ax, "B")

        ax = axes[2]
        for index, dataset in enumerate(("vanilla_syns", "specific_syns")):
            subset = [row for row in tracin if row["dataset"] == dataset]
            if not subset:
                mark_missing(ax, index)
                continue
            kept = sum(row["keep"] for row in subset)
            deleted = len(subset) - kept
            keep_percent = 100 * kept / len(subset)
            delete_percent = 100 - keep_percent
            ax.bar(index, keep_percent, color=KEEP_COLOR, edgecolor="black")
            ax.bar(index, delete_percent, bottom=keep_percent, color=DELETE_COLOR,
                   edgecolor="black", hatch="//")
            ax.text(index, keep_percent / 2, f"kept {kept}", ha="center", va="center",
                    fontsize=6.3, color="white" if keep_percent > 18 else "black")
            ax.text(index, keep_percent + delete_percent / 2, f"deleted {deleted}",
                    ha="center", va="center", fontsize=6.3)
        ax.set(xticks=(0, 1), xticklabels=("Question-\nagnostic", "Query-\nspecific"),
               ylim=(0, 100), ylabel="Candidate trajectories (%)",
               title="Frozen rule: summed TracIn > 0")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.legend(handles=[
            Patch(facecolor=KEEP_COLOR, edgecolor="black", label="Kept"),
            Patch(facecolor=DELETE_COLOR, edgecolor="black", hatch="//", label="Deleted"),
        ], loc="lower right", fontsize=6)
        panel_label(ax, "C")
        watermark(fig, complete)

        provenance = {
            "audience": "provisional ACL-style manuscript figure; publisher rules pending verification",
            "raw_data": [str(report_path), str(experiment_root / "cases" / "*/tracin/*.json")],
            "transformations": [
                "strict Oracle cohort: full-context Teacher wrong and initial KV wrong",
                "positive checkpoint influence: TracIn contribution > 0",
                "final retention: summed two-checkpoint TracIn score > frozen threshold 0",
            ],
            "uncertainty": "none displayed; exact candidate and paired-case proportions",
            "missing_data": "rendered as NA markers; sentinel insufficient-reference scores excluded from checkpoint sign rate",
            "complete": complete,
        }
        manifest = _export(export_figure, fig, output / "figure_selective_sd_diagnostics", provenance)
        plt.close(fig)
    return manifest


def write_latex_table(report, destination: Path):
    rows = accuracy_rows(report)
    overall = {row["arm"]: row for row in rows if row["scope"] == "overall"}
    lengths = {
        label: {row["arm"]: row for row in rows if row["scope_label"] == label}
        for label in ("8K", "16K", "32K", "64K")
    }
    def value(row):
        score = row["accuracy_percent"]
        return "--" if score is None else f"{score:.2f}"
    lines = [
        "% Generated from accuracy.json; missing values are rendered as --, never zero.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Four-arm BABILong accuracy on the same completed paired cases.}",
        "\\label{tab:contextual_cartridges_factorial}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Overall & 8K & 16K & 32K & 64K \\\\",
        "\\midrule",
    ]
    for arm in ARM_ORDER:
        name = ARM_LABELS[arm].replace("+", "$+$")
        cells = [value(overall[arm])] + [value(lengths[label][arm])
                                         for label in ("8K", "16K", "32K", "64K")]
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    destination.write_text("\n".join(lines) + "\n")


def render(report_path: Path, output: Path, *, experiment_root=None,
           allow_partial=False):
    import matplotlib
    matplotlib.use("Agg")
    report = read_json(report_path)
    complete, completed, planned = require_final_report(
        report, allow_partial=allow_partial)
    arm_names(report)
    experiment_root = Path(experiment_root) if experiment_root else report_path.parent.parent
    output.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_rows(report)
    contrasts = contrast_rows(report)
    oracle = oracle_rows(report)
    tracin = tracin_rows(experiment_root)
    if complete and not tracin:
        raise ValueError("complete four-arm report has no TracIn audit records")
    write_csv(output / "source_data_accuracy.csv", accuracy)
    write_csv(output / "source_data_paired_changes.csv", contrasts)
    write_csv(output / "source_data_oracle_escape.csv", oracle)
    write_csv(output / "source_data_tracin_candidates.csv", tracin)
    export_figure, style_context = _style_imports()
    main_export = main_results_figure(
        report, output, complete=complete, export_figure=export_figure,
        style_context=style_context, report_path=report_path)
    diagnostics_export = selective_diagnostics_figure(
        report, tracin, output, complete=complete, export_figure=export_figure,
        style_context=style_context, report_path=report_path,
        experiment_root=experiment_root)
    write_latex_table(report, output / "table_four_arm_accuracy.tex")
    descriptions = {
        "figure_main_results": (
            "Three-panel figure. Panel A reports exact BABILong accuracy for the four "
            "factorial arms on the same paired cases. Panel B reports the same metric "
            "by 8K, 16K, 32K, and 64K document length without connecting missing values. "
            "Panel C reports paired fixes to the right and regressions to the left, with "
            "a black diamond for net fixes minus regressions."
        ),
        "figure_selective_sd_diagnostics": (
            "Three-panel diagnostic. Panel A reports Selective-SD fixes and regressions "
            "within the strict cohort where both the full-context Teacher and initial KV "
            "were wrong. Panel B reports the fraction of candidate trajectories with "
            "positive signed influence at the initial and Vanilla-SD checkpoints. Panel C "
            "reports trajectories kept versus deleted by the frozen summed-TracIn-greater-"
            "than-zero rule."
        ),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "completed_cases": completed,
        "planned_cases": planned,
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "experiment_root": str(experiment_root),
        "artifacts": {
            "main_results": main_export,
            "selective_sd_diagnostics": diagnostics_export,
            "table": str(output / "table_four_arm_accuracy.tex"),
            "source_data": [
                str(output / "source_data_accuracy.csv"),
                str(output / "source_data_paired_changes.csv"),
                str(output / "source_data_oracle_escape.csv"),
                str(output / "source_data_tracin_candidates.csv"),
            ],
        },
        "descriptions": descriptions,
        "scientific_scope": (
            "fixed 160-case development cohort; no repeated-run uncertainty; "
            "figures do not by themselves establish statistical generalization"
        ),
        "missing_data_policy": "NA/gap, never zero",
        "software": {"matplotlib": matplotlib.__version__},
    }
    write_json(output / "figure_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = render(args.report, args.output,
                      experiment_root=args.experiment_root,
                      allow_partial=args.allow_partial)
    print(json.dumps({
        "complete": manifest["complete"],
        "completed_cases": manifest["completed_cases"],
        "planned_cases": manifest["planned_cases"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
