from contextlib import contextmanager, nullcontext
from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import masked_ttt as real_runtime
import experiments.anchored_short_verifier.paired_controls as real_controls
from experiments.gain_rate_pair import job, monitor, report, runner
from experiments.gain_rate_pair.runner import configs, pair_runner, real_pair_preflight
from experiments.gqa_head_gain.report import summarize
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from masked_ttt import Request
from masked_ttt.config import TTTConfig
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_materialization_equivalence import gpt_gqa_engine
from test_public_linked_selection import _source
from test_standalone import make_engine
from test_study_job import source_protocols, sources


@pytest.mark.parametrize("model,value_lr", [("qwen", .0001), ("gpt", .001)])
def test_configs_change_only_gain_rate(model, value_lr):
    baseline, candidate = configs(TTTConfig(), model)
    assert baseline.gain_learning_rate == .01 and candidate.gain_learning_rate == .03
    assert baseline.value_learning_rate == candidate.value_learning_rate == value_lr
    assert baseline.student_rollout_tokens == candidate.student_rollout_tokens == 8
    assert {f.name for f in fields(baseline) if getattr(baseline, f.name) != getattr(candidate, f.name)} == {
        "gain_learning_rate"}
    assert candidate.selected_transfer_weight == 0 and candidate.value_basis_source == "document"
    with pytest.raises(ValueError, match="Unknown model"):
        configs(TTTConfig(), "other")


def mock_driver():
    old_method, old_selector, old_controls, physical = object(), object(), object(), object()
    runtime = SimpleNamespace(MaskedTTT=old_method, FrozenJSONSelector=old_selector)
    controls = SimpleNamespace(original_question_controls=old_controls)
    def preflight(_):
        assert runtime.MaskedTTT is old_method and controls.original_question_controls is physical
        return dict(passed=True)
    impl = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x, validate=lambda *_: None,
                           replace=replace, preflight=preflight)
    return impl, runtime, controls, old_method, old_selector, old_controls, physical


def test_runner_restores_hooks_and_rejects_protocol_drift(monkeypatch):
    impl, runtime, controls, method, selector, ctrl, physical = mock_driver()
    monkeypatch.setattr(runner, "real_pair_preflight", lambda _: dict(paired_gain_rate_preflight=True))
    manifest = dict(mode="development", models=["qwen", "gpt"], selection_policy="decision_entity_link_v1",
        entity_reserve_fraction=.25, student_rollout_tokens=8, baseline_gain_lr=.01, candidate_gain_lr=.03,
        value_learning_rates={"qwen":.0001,"gpt":.001}, holdout_evaluated_at_selection=False)
    with pair_runner(impl, runtime, controls, root="new", selector="linked", physical_controls=physical):
        assert impl.ROOT == "new" and runtime.FrozenJSONSelector == "linked"
        assert controls.original_question_controls is nullcontext
        assert impl.validate_manifest(manifest)["selection_policy"] == "cached_decision_score10"
        assert impl.preflight(None)["paired_gain_rate_preflight"]
        with pytest.raises(ValueError, match="protocol changed"):
            impl.validate_manifest(dict(manifest, candidate_gain_lr=.1))
    assert (impl.ROOT, runtime.MaskedTTT, runtime.FrozenJSONSelector, controls.original_question_controls) == (
        "old", method, selector, ctrl)


@pytest.mark.parametrize("factory,model", [(make_engine,"qwen"), (gpt_gqa_engine,"gpt")])
def test_small_model_pair_materializes_and_releases_both_arms(tmp_path, factory, model):
    document, question, _, _ = _source(tmp_path)
    engine = factory()
    impl = SimpleNamespace(ROOT="old", validate_manifest=lambda x: x, validate=lambda *_: None,
                           replace=replace, preflight=lambda _: dict(passed=True))
    with pair_runner(impl, real_runtime, real_controls, root="candidate", selector=LinkedFrozenSelector,
                     physical_controls=kv_only_original_question_controls):
        engine.config = impl.replace(engine.config, max_context=engine.config.max_context, keep_fraction=.1,
            self_question_count=0, guard_enabled=False, selected_transfer_weight=0., value_basis_source="document",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        selector = real_runtime.FrozenJSONSelector(tmp_path)
        selector.bind_case("case")
        with real_controls.original_question_controls():
            result = real_runtime.MaskedTTT(engine, selector).answer(
                Request(document=document, question=question, max_new_tokens=3))
        pair = result["gain_rate_pair"]
        assert pair["model"] == model
        assert pair["same_selection"] and pair["same_teacher_tokens"] and pair["same_zero_tokens"]
        assert pair["baseline_standalone_released"] and pair["candidate_standalone_released"]
        assert pair["changed_settings"] == ["gain_learning_rate"]
        assert pair["baseline_gain_lr"] == .01 and pair["candidate_gain_lr"] == .03
        assert pair["value_lr"] == runner.VALUE_RATES[model]
        assert not pair["same_gpu"]
        with pytest.raises(ValueError, match="gain-rate arms changed"):
            impl.validate(result, {"key": "case"})
        # Synthetic metadata exercises the independent report validator; it is
        # never saved as a real-GPU or benchmark result.
        result.update(key="case", model_short=model, task="qa1", keep_fraction=.1)
        result["request_chat_date"] = "2026-11-03" if model == "gpt" else None
        result["self_judge"]["native_prediction"] = pair["baseline_native_prediction"] = "synthetic"
        pair["same_gpu"] = True
        report.validate_pair(result, "case", model, "qa1")
        pair["candidate_gain_lr"] = .01
        with pytest.raises(ValueError, match="gain-rate pair changed"):
            report.validate_pair(result, "case", model, "qa1")
    with pytest.raises(ValueError, match="allocated GPU"):
        real_pair_preflight(engine)


def test_manifest_and_batch_fix_two_model_gain_protocol():
    manifest = job.make_manifest(*source_protocols())
    assert manifest["models"] == ["qwen", "gpt"] and manifest["case_count"] == 200
    assert (manifest["baseline_gain_lr"], manifest["candidate_gain_lr"]) == (.01,.03)
    assert manifest["value_learning_rates"] == {"qwen":.0001,"gpt":.001}
    assert manifest["online_api_calls"] == 0 and manifest["reference_answers_used_in_inference"] is False
    for name in (job.DEV, *job.HOLDOUTS.values()):
        script = job.batch_script(Path("/tmp") / name)
        assert "#SBATCH --gpus-per-task=1" in script and "#SBATCH --mem=64G" in script
        assert "gain_rate_pair/execute.py" in script and "unset OPENAI_API_KEY" in script
        assert 'TRITON_CACHE_DIR="$TASK_ROOT/runtime_cache/triton"' in script
        assert 'TORCH_EXTENSIONS_DIR="$TASK_ROOT/runtime_cache/torch_extensions"' in script
    bad = list(source_protocols())
    bad[2]["gain_learning_rate"] = .1
    with pytest.raises(ValueError, match="baseline changed"):
        job.make_manifest(*bad)


def make_rows(cases):
    return [dict(case=case, task="qa1", length=8192, whole=0, zero=0, teacher=0,
                 baseline=0, cartridge=1, cartridge_percent=10.) for case in cases]


def test_holdout_gate_is_model_specific_and_requires_all_controls(tmp_path):
    manifest = job.make_manifest(*source_protocols())
    development = tmp_path / job.DEV
    (development / "runtime").mkdir(parents=True)
    (development / "runtime/frozen.txt").write_text("unchanged")
    (development / "manifest.json").write_text(json.dumps(manifest))
    rows = make_rows(manifest["case_ids"])
    data = dict(mode="development", expected_per_model=200, metric="answer_focused_lexical_v2",
        errors=[], invalid=[], models={"qwen":dict(overall=summarize(rows), rows=rows)})
    root, decision = monitor.prepare_holdout(tmp_path, development, data, "qwen")
    assert decision["eligible"] and root.name == job.HOLDOUTS["qwen"]
    assert json.loads((root / "manifest.json").read_text())["models"] == ["qwen"]
    assert (root / "runtime/frozen.txt").read_text() == "unchanged"
    assert not (root / "submission.json").exists()
    root2, again = monitor.prepare_holdout(tmp_path, development, data, "qwen")
    assert root2 == root and again == decision
    no_gain = make_rows(manifest["case_ids"])
    for row in no_gain:
        row["teacher"] = 1
    data["models"]["qwen"] = dict(overall=summarize(no_gain), rows=no_gain)
    assert not monitor.development_decision(monitor.model_view(data,"qwen"))["eligible"]
    incomplete = monitor.model_view(data,"qwen")
    incomplete["overall"]["n"] = 199
    assert not monitor.development_decision(incomplete)["eligible"]


def test_combined_report_requires_source_disjoint_complete_cases():
    dev, held = sources()
    manifest = dict(models=["qwen"], development_case_ids=dev, holdout_case_ids=held,
                    qualitative_case_sources=["qa1-300023","qa2-300037"])
    development = dict(metric="answer_focused_lexical_v2", errors=[], invalid=[], rows=make_rows(dev))
    holdout = dict(mode="holdout", expected_per_model=800, metric="answer_focused_lexical_v2", errors=[],
        invalid=[], models={"qwen":dict(overall=summarize(make_rows(held)), rows=make_rows(held))})
    data = monitor.combine(manifest, development, holdout)
    assert data["models"]["qwen"]["overall"]["n"] == 1000
    assert data["holdout_excluding_two_previously_viewed_sources"]["n"] == 792
    holdout["models"]["qwen"]["rows"].pop()
    with pytest.raises(ValueError, match="Complete"):
        monitor.combine(manifest, development, holdout)


def test_report_validates_gain_before_scoring(tmp_path, monkeypatch):
    manifest = job.make_manifest(*source_protocols())
    case = manifest["case_ids"][0]
    folder = tmp_path / "cases/qwen" / case
    folder.mkdir(parents=True)
    result = dict(length=8192, self_judge=dict(native_prediction="wrong"),
        teacher=dict(prediction="right"), prediction="right",
        gain_rate_pair=dict(baseline_prediction="wrong"),
        standalone_cartridge=dict(original_question_controls=dict(zero=dict(prediction="wrong")),
                                  memory=dict(standalone_cartridge_bytes=100)),
        memory=dict(full_document_kv_bytes=1000))
    (folder / "result.json").write_text(json.dumps(result))
    seen = []
    monkeypatch.setattr(report, "validate_pair", lambda item, actual_case, model, task:
                        seen.append((actual_case,model,task)))
    references = {case:dict(task="qa1", answers=["right"])}
    score = lambda text, answers, *_: (text in answers,None)
    data = report.collect(manifest,tmp_path,references,score)
    assert seen == [(case,"qwen","qa1")] and data["models"]["qwen"]["overall"]["n"] == 1
    assert data["models"]["qwen"]["overall"]["strictly_paired"]["new_correct"] == 1
    assert data["models"]["gpt"]["overall"]["n"] == 0
    assert not data["errors"] and not data["invalid"]


def test_submit_is_one_gpu_and_idempotent(tmp_path, monkeypatch):
    root = tmp_path / job.DEV
    root.mkdir()
    (root / "cpu_validation_passed.json").write_text(json.dumps(dict(cpu_tests_passed=True)))
    (root / "job.sbatch").write_text("# test\n")
    (root / "manifest.json").write_text(json.dumps(job.make_manifest(*source_protocols())))
    @contextmanager
    def available():
        yield True
    monkeypatch.setitem(sys.modules,"gpu_slot_budget",SimpleNamespace(submission_slot=available))
    calls = []
    monkeypatch.setattr(job.subprocess,"run",lambda args,**_kwargs:
                        calls.append(args) or SimpleNamespace(stdout="123\n"))
    sent = job.submit(tmp_path,root)
    assert sent["job_id"] == "123" and sent["gpus"] == 1 and sent["models"] == ["qwen","gpt"]
    assert job.submit(tmp_path,root)["job_id"] == "123" and len(calls) == 1
