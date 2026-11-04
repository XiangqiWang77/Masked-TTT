import json

import pytest

from experiments.value_lr_physical import freeze_final_gain_transfer as freeze


def test_freezes_only_complete_grouped_readouts_and_applies_denominator_filter(tmp_path, monkeypatch):
    cases = [f"babilong-{length}-qa1-{i}" for i in range(10) for length in ("8", "16", "32", "64")]
    (tmp_path / "manifest.json").write_text(json.dumps(dict(case_ids=cases)))
    for model in ("qwen", "gpt"):
        folder = tmp_path / "cases" / model
        folder.mkdir(parents=True)
        for i, case in enumerate(cases):
            readout = dict(kl_full_native_to_full_adapted=2e-8 if i < 20 else 1e-9,
                           kl_selected_native_to_selected_adapted=1e-8)
            (folder / (case + ".json")).write_text(json.dumps(dict(readout=readout)))
    overall = dict(cases=40, readouts=40, effect_ratio_count=20, closer_to_full=10,
        farther_from_full=30, median_first_gradient_common_squared_fraction=.1,
        median_first_update_common_squared_fraction=.2, update_common_over_90_percent=0, nonzero_updates=40)
    monkeypatch.setattr(freeze, "collect", lambda _: dict(models={
        model: dict(overall=overall) for model in ("qwen", "gpt")}))
    result = freeze.freeze(tmp_path)
    assert result["models"]["qwen"]["ratio_count"] == 20
    assert result["models"]["qwen"]["median_selected_over_full_first_token_kl"] == .5
    assert result["quality_or_correctness_not_measured"] and not result["reference_answers_used"]
    overall["readouts"] = 39
    with pytest.raises(ValueError, match="all 40"):
        freeze.freeze(tmp_path)
