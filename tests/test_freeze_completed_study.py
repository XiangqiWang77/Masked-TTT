from collections import defaultdict
from copy import deepcopy
import json

import pytest

from experiments.gqa_head_gain.report import summarize
from experiments.self_study_prompt.freeze_completed_study import freeze, render
from experiments.self_study_prompt.study_job import make_manifest
from test_study_job import source_protocols


def records():
    manifest = make_manifest(*source_protocols())
    rows = []
    for i, case in enumerate(manifest["case_ids"]):
        _, length, source = case.split("-", 2)
        rows.append(dict(case=case, task=source.split("-", 1)[0], length=int(length)*1024,
            whole=0, zero=0, teacher=1, baseline=0, cartridge=int(i == 0), cartridge_percent=10.))
    groups = defaultdict(list)
    for row in rows:
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    report = dict(mode="development", expected=200, metric="answer_focused_lexical_v2",
        online_api_calls=0, reference_answers_used_only_posthoc=True, errors=[], invalid=[], rows=rows,
        overall=summarize(rows), groups={k: summarize(v) for k, v in groups.items()},
        practice=dict(original=2, study_only=3, more=1, fewer=0, identical_checks=199))
    probe = dict(state="complete", n=10, expected=10, errors=0, reference_answers_used=False,
        online_api_calls=0, training_runtime_changed=False, same_selected_teacher_cache=True,
        downstream_accuracy_not_measured=True, base=dict(n=10, verified_exercises=0),
        variant=dict(n=10, verified_exercises=2), more_verified=1, fewer_verified=0)
    preflight = dict(passed=True, paired_study_preflight=True, same_gpu=True,
                     independently_materialized_and_released=True, gpu="synthetic")
    finished = dict(all_complete=True, completed=dict(qwen=200))
    submission = dict(gpus=1, cases=200, online_api_calls=0, job_id="123")
    return manifest, report, probe, preflight, finished, submission


def test_freeze_aggregates_without_case_ids():
    inputs = records()
    data = freeze(*inputs)
    assert data["completed_cases"] == 200 and data["probe_decision"]["eligible"]
    assert not data["development_decision"]["eligible"]
    assert data["overall"]["strictly_paired"]["new_correct"] == 1
    assert data["source_questions"] == 50 and data["group_source_questions"]["task:qa1"] == 50
    assert inputs[0]["case_ids"][0] not in json.dumps(data)
    assert "新答对 1 题" in render(data)
    assert "| QA1 | 200 | 50 |" in render(data)


def test_rejects_inconsistent_scores_and_failed_preflight():
    inputs = list(records())
    inputs[1] = deepcopy(inputs[1])
    inputs[1]["rows"][0]["cartridge"] = 0
    with pytest.raises(ValueError, match="aggregates"):
        freeze(*inputs)
    inputs = list(records())
    inputs[3] = dict(inputs[3], same_gpu=False)
    with pytest.raises(ValueError, match="same-GPU"):
        freeze(*inputs)
