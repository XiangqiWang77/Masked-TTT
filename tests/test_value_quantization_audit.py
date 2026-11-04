import pytest

from experiments.babilong_value_lr.audit_quantization import RATES, summarize


def test_quantization_summary_uses_paired_results_and_normed_error():
    row = dict(case="babilong-8-qa1-1", zero=0, teacher=0)
    for index, rate in enumerate(RATES):
        row.update({rate: int(index > 0), rate+"_changed": .25, rate+"_relative_l2": .001,
                    rate+"_error": .6})
    stats = summarize([row])
    assert stats["n"] == 1 and stats["source_questions"] == 1
    assert stats["rates"][RATES[0]]["median_squared_write_error_percent"] == 60
    assert stats["rates"][RATES[1]]["versus_zero"]["new_correct"] == 1
    row[RATES[0]+"_error"] = float("nan")
    with pytest.raises(ValueError):
        summarize([row])
