from experiments.report_syn_sd_factorial import aggregate


ARMS = (
    "vanilla_syns_vanilla_sd",
    "vanilla_syns_selective_sd",
    "specific_syns_vanilla_sd",
    "specific_syns_selective_sd",
)


def row(case, vanilla, vanilla_selective, specific, specific_selective):
    return {
        "case": case,
        "scores": dict(zip(
            ARMS,
            (vanilla, vanilla_selective, specific, specific_selective),
        )),
        "baseline_scores": {"full_context_teacher": 0, "initial_kv": 0},
        "training": {
            arm: {"updates": 2, "all_rejected_original_SD_fallback": False,
                  "all_rejected_validation_original_SD_fallback": False}
            for arm in ARMS
        },
    }


def test_report_uses_vanilla_sd_names_and_counts_strict_oracle_escape_net():
    rows = [
        row("fix-1", 0, 1, 0, 1),
        row("regress", 1, 0, 1, 0),
        row("fix-2", 0, 1, 0, 1),
    ]
    result = aggregate(rows, ARMS, {"official_epochs": 2})

    assert result["contrasts"]["query_specific_with_vanilla_SD"]["n"] == 3
    assert result["contrasts"][
        "contextual_cartridges_vs_vanilla_cartridges"
    ]["delta_pp"] == 100 / 3
    strict = result["oracle_trap_evidence"]["specific_syns"]
    assert strict["strict_oracle_escape_cohort"] == 3
    assert strict["strict_oracle_escape_fixes"] == 2
    assert strict["strict_oracle_escape_regressions"] == 1
    assert strict["strict_oracle_escape_net"] == 1
