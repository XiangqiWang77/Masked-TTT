from pathlib import Path
import shutil
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from experiments.kv_only_cartridge import enable_in_job_probe as setup


def workspace(tmp_path):
    for name in setup.TARGETS:
        root = tmp_path / name
        runtime = root / "runtime"
        runtime.mkdir(parents=True)
        (runtime / "rollout_driver.py").write_text("precision = 'disabled_optional_probe'\n")
        destination = runtime / "repo/src/masked_ttt"
        destination.parent.mkdir(parents=True)
        shutil.copytree(setup.REPO / "src/masked_ttt", destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return tmp_path


def test_enable_is_idempotent_and_keeps_frozen_training_unchanged(tmp_path):
    root = workspace(tmp_path)
    frozen = root / setup.TARGETS[0] / "runtime/repo/src/masked_ttt/pipeline.py"
    before = frozen.read_bytes()
    result = setup.enable(root)
    assert result["enabled"] and not result["training_runtime_changed"] and result["new_gpu_jobs"] == 0
    assert frozen.read_bytes() == before
    central = Path(result["shared_results"])
    assert (central / "pipeline_candidate.py").is_file() and (central / "ready.json").is_file()
    for target in setup.TARGETS:
        option = root / target / "disabled_optional_probe"
        assert (option / "driver.py").is_file() and (option / "ready.json").is_file()
    (central / "qwen.json").write_text('{"passed":true}')
    assert setup.enable(root) == result


def test_refuses_runtime_mismatch_before_enabling(tmp_path):
    root = workspace(tmp_path)
    frozen = root / setup.TARGETS[0] / "runtime/repo/src/masked_ttt/attention.py"
    frozen.write_text("different runtime\n")
    with pytest.raises(ValueError, match="supporting runtime module differs"):
        setup.enable(root)
    assert not (root / setup.CENTRAL).exists()


def test_refuses_unrelated_optional_directory(tmp_path):
    root = workspace(tmp_path)
    option = root / setup.TARGETS[0] / "disabled_optional_probe"
    option.mkdir()
    (option / "driver.py").write_text("other experiment\n")
    with pytest.raises(ValueError, match="different configuration"):
        setup.enable(root)
    assert (option / "driver.py").read_text() == "other experiment\n"


def test_deployed_wrapper_uses_the_shared_probe(tmp_path):
    import json
    from test_standalone import make_engine

    root = workspace(tmp_path)
    result = setup.enable(root)
    path = root / setup.TARGETS[0] / "disabled_optional_probe/driver.py"
    spec = spec_from_file_location("optional_probe_wrapper", path)
    module = module_from_spec(spec);spec.loader.exec_module(module)
    assert module.run_if_needed(make_engine(), "qwen", {"gpu": "cpu-test"}) == "passed"
    data = json.loads((Path(result["shared_results"]) / "qwen.json").read_text())
    assert data["same_standalone_logits"] and data["original_benchmark_runtime_unchanged"]
