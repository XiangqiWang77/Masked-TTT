"""Exercise the real frozen reporter and 56-to-200 case reuse on synthetic data."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import experiments.anchored_short_verifier.paired_controls as paired_controls
from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
from experiments.long_rollout_steps.extend import prepare
from experiments.long_rollout_steps.prepare import report_stub
from experiments.long_rollout_steps.runner import long_steps_runner
from masked_ttt import MaskedTTT, Request
from masked_ttt.linked_selection import LinkedFrozenSelector
import masked_ttt as runtime
from test_long_rollout_steps_extend import fixture_protocol
from test_public_linked_selection import _source
from test_standalone import make_engine


def test_extension_reuses_real_reporter_validated_pilot_results(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pilot_root = workspace / "query_cartridges_longsteps_pilot_20261104"
    dev_root = workspace / "query_cartridges_longsteps_200dev_20261104"
    pilot_root.mkdir()
    frozen = pilot_root / "runtime/repo"
    frozen.mkdir(parents=True)
    for folder in ("src", "experiments", "tests"):
        shutil.copytree(repo / folder, frozen / folder,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    scorer = pilot_root / "runtime/scorer"
    scorer.mkdir()
    (scorer / "report_hybrid_partial.py").write_text(
        "BABILONG_SCORER_VERSION='answer_focused_lexical_v2'\n"
        "def score_prediction(prediction, references, benchmark, task):\n"
        "    return (int(bool(prediction)), True)\n")
    pilot, _ = fixture_protocol()
    (pilot_root / "manifest.json").write_text(json.dumps(pilot))
    references = {case: {"task": case.split("-", 3)[2], "answers": ["blue"]}
                  for case in pilot["case_ids"]}
    ref_dir = tmp_path / "chunked_ntp_query_only_20261029"
    ref_dir.mkdir()
    (ref_dir / "evaluation_references.json").write_text(json.dumps(references))

    selector_root = tmp_path / "selector"
    selector_root.mkdir()
    document, question, _, _ = _source(selector_root)
    engine = make_engine()
    engine.eos_ids = {-1}
    implementation = SimpleNamespace(ROOT="old", validate_manifest=lambda value: value,
        validate=lambda _result, _entry: None, replace=replace, preflight=lambda _engine: None)
    with long_steps_runner(implementation, runtime, paired_controls, root="synthetic-pilot",
            linked_selector=LinkedFrozenSelector, physical_controls=kv_only_original_question_controls):
        engine.config = implementation.replace(engine.config, max_context=2048, keep_fraction=.1,
            student_rollout_tokens=32, self_question_count=0, guard_enabled=False,
            selected_transfer_weight=0., value_basis_source="document",
            release_full_before_cartridge=True, release_selected_source_before_decode=True)
        selector = runtime.FrozenJSONSelector(tmp_path / "selector")
        selector.bind_case("case")
        with paired_controls.original_question_controls():
            sample = MaskedTTT(engine, selector).answer(Request(
                document=document, question=question, max_new_tokens=4))
        implementation.validate(sample, {"key": "case"})
    sample["self_judge"]["native_prediction"] = "blue"
    for model, name in (("qwen", "Qwen/Qwen3-4B-Instruct-2507"), ("gpt", "openai/gpt-oss-20b")):
        for case in pilot["case_ids"]:
            record = deepcopy(sample)
            record.update(key=case, model=name, task=references[case]["task"], keep_fraction=.1,
                          length=int(case.split("-")[1]), request_chat_date="2026-11-03")
            record["selection"]["source_case_key"] = case
            path = pilot_root / "cases" / model / case / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record))
    (pilot_root / "report.py").write_text(report_stub(workspace, pilot_root))
    subprocess.run([sys.executable, str(pilot_root / "report.py")], check=True,
                   capture_output=True, text=True, timeout=60)
    (pilot_root / "finished.json").write_text('{"all_complete":true}\n')
    prepared = prepare(workspace, pilot_root, dev_root)
    assert prepared["case_count"] == 200 and prepared["new_cases_per_model"] == 144
    report = json.loads((dev_root / "latest.json").read_text())
    assert not report["invalid"] and not report["errors"]
    assert all(report["models"][model]["overall"]["n"] == 56 for model in ("qwen", "gpt"))
    case = pilot["case_ids"][0]
    assert (dev_root / "cases/qwen" / case / "result.json").read_bytes() == (
        pilot_root / "cases/qwen" / case / "result.json").read_bytes()
