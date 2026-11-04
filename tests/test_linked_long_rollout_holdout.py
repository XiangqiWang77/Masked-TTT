import json

from experiments.babilong_linked_selection.long_rollout_holdout import freeze
from experiments.babilong_linked_selection.long_rollout_protocol import development_decision, summarize
from test_linked_long_rollout_report import fixture_manifest


def development(tmp_path, qwen_wins=True, complete=True):
    root = tmp_path / "query_cartridges_linked_roll32_dev_20261104"
    root.mkdir()
    runtime = root / "runtime"
    for directory in ("repo/src/masked_ttt", "repo/experiments", "repo/tests"):
        (runtime / directory).mkdir(parents=True)
    for filename in ("rollout_driver.py", "olmo_profile.json", "helpers/time_budget.py",
                     "scorer/report_hybrid_partial.py", "scorer/report_qwen_longbench_partial.py",
                     "gpu-tests/test_cuda_materialization.py"):
        path = runtime / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    manifest = fixture_manifest()
    manifest["models"] = ["qwen", "gpt"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    models = {}
    for model in manifest["models"]:
        wins = model == "qwen" and qwen_wins
        rows = [dict(case=case, whole=0, zero=0, teacher=0 if wins else 1,
                     cartridge=1, linked8=0) for case in manifest["case_ids"]]
        if not complete:
            rows.pop()
        total = summarize(rows)
        models[model] = {"overall": total, "rows": rows, "decision": development_decision(total)}
    report = {"mode": "development", "expected_per_model": 200, "metric": "answer_focused_lexical_v2",
        "selection_policy": "decision_entity_link_v1", "cached_scores": True,
        "student_rollout_tokens": 32, "selected_transfer_weight": 0,
        "evaluation_answers_used_only_posthoc": True, "online_api_calls": 0,
        "errors": [], "invalid": [], "models": models}
    (root / "latest.json").write_text(json.dumps(report))
    return root


def test_holdout_waits_without_a_finished_development(tmp_path):
    root = development(tmp_path)
    assert freeze(tmp_path, root) is None
    assert not (tmp_path / "query_cartridges_linked_roll32_holdout_20261104").exists()


def test_holdout_freezes_only_qualified_model_and_does_not_mutate_snapshot(tmp_path):
    root = development(tmp_path)
    (root / "finished.json").write_text('{"all_complete": true}')
    prepared = freeze(tmp_path, root)
    assert prepared["state"] == "prepared" and prepared["manifest"]["models"] == ["qwen"]
    holdout = prepared["root"]
    assert len(prepared["manifest"]["case_ids"]) == 800
    assert "#SBATCH --job-name=qc-linked-roll32-hold" in (holdout / "job.sbatch").read_text()
    (root / "latest.json").write_text("{}")
    assert freeze(tmp_path, root)["manifest"] == prepared["manifest"]
    assert not (holdout / "submission.json").exists()


def test_no_holdout_if_incomplete_or_no_model_exceeds_controls(tmp_path):
    root = development(tmp_path, complete=False)
    (root / "finished.json").write_text('{"all_complete": true}')
    assert freeze(tmp_path, root) is None
    assert not (tmp_path / "query_cartridges_linked_roll32_holdout_20261104/development_snapshot.json").exists()


def test_no_eligible_model_produces_no_gpu_job(tmp_path):
    root = development(tmp_path, qwen_wins=False)
    (root / "finished.json").write_text('{"all_complete": true}')
    outcome = freeze(tmp_path, root)
    assert outcome["state"] == "not_eligible"
    assert (outcome["root"] / "no_eligible_model.json").exists()
    assert not (outcome["root"] / "submission.json").exists()
