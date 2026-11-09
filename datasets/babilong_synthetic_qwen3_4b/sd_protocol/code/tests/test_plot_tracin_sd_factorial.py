import json

import pytest

from experiments.plot_tracin_sd_factorial import (
    ARM_ORDER,
    render,
    require_final_report,
)
from experiments.report_syn_sd_factorial import aggregate


def case(name, scores, teacher=0, initial=0):
    return {
        "case": name,
        "scores": dict(zip(ARM_ORDER, scores)),
        "baseline_scores": {
            "full_context_teacher": teacher,
            "initial_kv": initial,
        },
        "training": {
            arm: {
                "updates": 2,
                "all_rejected_original_SD_fallback": False,
                "all_rejected_validation_original_SD_fallback": False,
            }
            for arm in ARM_ORDER
        },
    }


def fixture_report():
    rows = [
        case("qa1_8k_0", (0, 1, 0, 1)),
        case("qa2_16k_0", (1, 1, 1, 1), teacher=1, initial=1),
        case("qa3_32k_0", (1, 0, 1, 1)),
        case("qa4_64k_0", (0, 1, 1, 1)),
    ]
    config = {"official_epochs": 2}
    overall = aggregate(rows, ARM_ORDER, config)
    by_length = {}
    for length, row in zip((8192, 16384, 32768, 65536), rows):
        by_length[str(length)] = aggregate([row], ARM_ORDER, config)
    return {
        "planned": 4,
        "matrix": list(ARM_ORDER),
        "overall": overall,
        "by_length": by_length,
        "complete": True,
    }


def write_audit(root, case_name, dataset):
    path = root / "cases" / case_name / "tracin" / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "checkpoints": [
            {"checkpoint": "initial_cartridge"},
            {"checkpoint": "vanilla_SD"},
        ],
        "decisions": [
            {
                "candidate_index": 0,
                "tracin_score": 0.5,
                "threshold": 0.0,
                "keep": True,
                "checkpoint_contributions": [0.4, 0.1],
            },
            {
                "candidate_index": 1,
                "tracin_score": -0.2,
                "threshold": 0.0,
                "keep": False,
                "checkpoint_contributions": [0.1, -0.3],
            },
        ],
    }))


def test_final_renderer_refuses_incomplete_report():
    report = fixture_report()
    report["complete"] = False
    with pytest.raises(ValueError, match="incomplete report"):
        require_final_report(report)


def test_renderer_exports_figures_tables_and_source_data(tmp_path):
    report = fixture_report()
    report_path = tmp_path / "report" / "accuracy.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps(report))
    for case_name in ("qa1_8k_0", "qa2_16k_0", "qa3_32k_0", "qa4_64k_0"):
        for dataset in ("vanilla_syns", "specific_syns"):
            write_audit(tmp_path, case_name, dataset)

    output = tmp_path / "figures"
    manifest = render(report_path, output, experiment_root=tmp_path)

    assert manifest["complete"] is True
    for stem in ("figure_main_results", "figure_selective_sd_diagnostics"):
        for suffix in ("pdf", "svg", "png", "export.json"):
            path = output / f"{stem}.{suffix}"
            assert path.is_file() and path.stat().st_size > 0
    for name in (
        "source_data_accuracy.csv",
        "source_data_paired_changes.csv",
        "source_data_oracle_escape.csv",
        "source_data_tracin_candidates.csv",
        "table_four_arm_accuracy.tex",
        "figure_manifest.json",
    ):
        assert (output / name).is_file()
