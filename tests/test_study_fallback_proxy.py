from experiments.self_study_prompt.freeze_fallback_proxy import summarize


def test_proxy_only_substitutes_when_original_has_no_verified_practice():
    rows = [dict(case="case-8-a", task="qa1", baseline=0, cartridge=1, teacher=1,
                 baseline_verified=0, candidate_verified=2),
            dict(case="case-16-b", task="qa2", baseline=1, cartridge=0, teacher=1,
                 baseline_verified=1, candidate_verified=1),
            dict(case="case-32-c", task="qa3", baseline=1, cartridge=0, teacher=1,
                 baseline_verified=0, candidate_verified=0)]
    result = summarize(rows)
    assert result["cases_with_candidate_fallback"] == 1
    assert result["potential_additional_verified_practices"] == 2
    assert result["fallback_cases_candidate_vs_original"] == {"new_correct": 1, "new_wrong": 0}
    assert result["original_correct"] == 2 and result["proxy_correct"] == 3
