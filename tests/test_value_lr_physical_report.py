import pytest

from experiments.babilong_linked_selection.long_rollout_protocol import holdout_decision, summarize
from experiments.value_lr_physical.protocol import holdout_manifest
from experiments.value_lr_physical.report import collect, combine, render
from test_value_lr_physical import fixture_value_development


def fixture_report():
    development, report, profile = fixture_value_development()
    for row in report["models"]["gpt"]["rows"]:
        row.update(task=row["case"].split("-", 3)[2], length=int(row["case"].split("-")[1]))
    manifest = holdout_manifest(development, report, profile, "gpt")
    rows = [dict(case=case, task=case.split("-", 3)[2], length=int(case.split("-")[1]),
                 whole=int(i < 100), zero=int(i < 100), teacher=int(i < 100),
                 cartridge=int(i < 140), linked8=None, cartridge_percent=10.)
            for i, case in enumerate(manifest["case_ids"])]
    strict = [row for row in rows if row["case"].split("-", 2)[2] not in {"qa1-300023", "qa2-300037"}]
    holdout = dict(mode="holdout", model="gpt", fixed_value_learning_rate=.001, expected=800,
        metric="answer_focused_lexical_v2", overall=summarize(rows), groups={}, rows=rows,
        strict_holdout=summarize(strict), decision=holdout_decision(summarize(strict)), errors=[], invalid=[])
    return development, report, manifest, holdout


def test_empty_fixed_lr_holdout_stays_partial(tmp_path):
    _, _, manifest, _ = fixture_report()
    result = collect(manifest, tmp_path, {}, lambda *args: (0, True))
    assert result["model"] == "gpt" and result["expected"] == 800 and result["overall"]["n"] == 0


def test_full_1000_case_report_uses_development_fixed_rate():
    development, report, manifest, holdout = fixture_report()
    result = combine(development, report, manifest, holdout)
    assert result["overall"]["n"] == 1000 and result["development"]["n"] == 200
    assert result["holdout"]["n"] == 800 and result["strict_holdout"]["n"] == 792
    assert result["fixed_value_learning_rate"] == .001
    assert "ICL Teacher" in render(result)


def test_holdout_cannot_change_the_fixed_rate():
    development, report, manifest, holdout = fixture_report()
    manifest["fixed_value_learning_rate"] = .0003
    holdout["fixed_value_learning_rate"] = .0003
    with pytest.raises(ValueError, match="development-fixed"):
        combine(development, report, manifest, holdout)
