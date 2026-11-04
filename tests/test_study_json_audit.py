import pytest

from experiments.truncated_study_json.audit_development import summarize


def test_counts_limit_and_short_failures_separately():
    rows = [dict(generated_tokens=4, verified=0, rejections={"invalid_json": 1}),
            dict(generated_tokens=192, verified=0, rejections={"invalid_json": 1}),
            dict(generated_tokens=192, verified=1, rejections={})]
    result = summarize(rows, 192)
    assert result["invalid_json_cases"] == 2
    assert result["invalid_json_at_token_limit"] == 1
    assert result["invalid_json_below_token_limit"] == 1
    assert result["cases_with_verified_practice"] == 1


def test_rejects_invalid_generation_lengths():
    with pytest.raises(ValueError):
        summarize([dict(generated_tokens=193, verified=0, rejections={})], 192)
