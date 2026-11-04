from collections import defaultdict
import json

import pytest

from experiments.gqa_head_gain.report import summarize
from experiments.linked_value.freeze_development import collect
from test_canonical_memory_report import result as memory_result
from test_study_job import sources


def fixture(root):
    keys, _ = sources()
    manifest = dict(mode="development", models=["gpt"], case_count=200, case_ids=keys,
        selection_policy="decision_entity_link_v1", fixed_value_learning_rate=.001,
        student_rollout_tokens=8, selected_transfer_weight=0,
        reference_answers_used_in_inference=False, online_api_calls=0)
    rows, groups = [], defaultdict(list)
    for key in keys:
        length = int(key.split("-", 2)[1]) * 1024
        row = dict(case=key, task="qa1", length=length, whole=0, zero=0, teacher=0,
                   cartridge=1, baseline=0, cartridge_percent=10.)
        rows.append(row)
        groups["task:qa1"].append(row)
        groups["length:" + str(length)].append(row)
        saved = memory_result()
        saved.update(key=key, model_short="gpt", task="qa1", length=length,
            training=dict(value_learning_rate=.001, gain_learning_rate=.01, selected_transfer_weight=0,
                          objective="on-policy KL(teacher || student)"),
            selection=dict(selection_policy="decision_entity_link_v1", reference_answers_used=False,
                           online_api_calls=0))
        folder = root / "cases/gpt" / key
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(saved))
    report = dict(mode="development", expected=200, fixed_value_learning_rate=.001,
        metric="answer_focused_lexical_v2", rows=rows, errors=[], invalid=[], unmatched_baselines=[],
        overall=summarize(rows), groups={key: summarize(value) for key, value in sorted(groups.items())})
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "latest.json").write_text(json.dumps(report))
    (root / "finished.json").write_text(json.dumps(dict(all_complete=True, completed={"gpt": 200})))
    return manifest, report


def test_exports_validated_paired_scores_and_physical_cache_aggregates(tmp_path):
    manifest, _ = fixture(tmp_path)
    output = collect(tmp_path)
    assert output["overall"]["n"] == 200 and output["overall"]["versus_teacher"]["delta_pp"] == 100
    assert output["memory"]["median_cartridge_percent_of_full"] == 10
    assert output["new_model_calls"] == 0 and output["online_api_calls"] == 0
    assert manifest["case_ids"][0] not in json.dumps(output)


def test_rejects_altered_groups_and_unreleased_cache(tmp_path):
    manifest, report = fixture(tmp_path)
    report["groups"]["task:qa1"]["n"] = 199
    (tmp_path / "latest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="aggregates"):
        collect(tmp_path)
    report["groups"]["task:qa1"]["n"] = 200
    (tmp_path / "latest.json").write_text(json.dumps(report))
    path = tmp_path / "cases/gpt" / manifest["case_ids"][0] / "result.json"
    saved = json.loads(path.read_text())
    saved["standalone_cartridge"]["memory"]["additional_source_kv_bytes_before_decode"] = 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="released"):
        collect(tmp_path)
