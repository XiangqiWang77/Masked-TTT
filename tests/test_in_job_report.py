import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from experiments.kv_only_cartridge import in_job_report
from experiments.kv_only_cartridge.in_job_report import REQUIRED, collect


def test_waiting_failed_passed_and_cpu_results_are_distinct(tmp_path):
    (tmp_path / "ready.json").write_text('{}')
    assert collect(tmp_path)["models"]["qwen"]["state"] == "waiting"
    data = dict(passed=True, model="qwen", model_type="qwen3", identity={"gpu": "NVIDIA H200"},
                online_api_calls=0, reference_answers_used=False, generic_teacher_answer_in_prompt=False, attempts=1,
                **{key: True for key in REQUIRED})
    (tmp_path / "qwen.json").write_text(json.dumps(data))
    (tmp_path / "gpt.json").write_text(json.dumps(dict(passed=False, error="ValueError", attempts=1)))
    result = collect(tmp_path)
    assert result["models"]["qwen"]["state"] == "passed"
    assert result["models"]["gpt"] == dict(state="failed", error_type="ValueError", attempts=1)
    assert result["scheduler_queries"] == 0 and result["new_gpu_jobs"] == 0
    data["identity"]["gpu"] = "cpu-test"
    (tmp_path / "qwen.json").write_text(json.dumps(data))
    assert collect(tmp_path)["models"]["qwen"]["state"] == "invalid"


def test_version_change_does_not_report_old_preflight_as_new(tmp_path):
    (tmp_path / "ready.json").write_text('{"probe_version": 4}')
    old = dict(passed=True, probe_version=3, model="qwen", model_type="qwen3",
               identity={"gpu": "H200"}, online_api_calls=0, reference_answers_used=False,
               generic_teacher_answer_in_prompt=False, **{key: True for key in REQUIRED})
    (tmp_path / "qwen.json").write_text(json.dumps(old))
    assert collect(tmp_path)["models"]["qwen"]["state"] == "waiting"
    old["probe_version"] = 4
    (tmp_path / "qwen.json").write_text(json.dumps(old))
    assert collect(tmp_path)["models"]["qwen"]["state"] == "passed"
    old["generic_teacher_answer_in_prompt"] = True
    (tmp_path / "qwen.json").write_text(json.dumps(old))
    assert collect(tmp_path)["models"]["qwen"]["state"] == "invalid"


def test_two_hourly_services_can_write_the_same_report(tmp_path):
    (tmp_path / "ready.json").write_text('{}')
    command = [sys.executable, str(Path(in_job_report.__file__)), "--directory", str(tmp_path)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: subprocess.run(command, capture_output=True, text=True), range(2)))
    assert all(result.returncode == 0 for result in results)
    assert json.loads((tmp_path / "latest.json").read_text())["models"]["qwen"]["state"] == "waiting"
