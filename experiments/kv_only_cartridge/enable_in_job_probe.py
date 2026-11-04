#!/usr/bin/env python3
"""Enable one optional real-model check in existing jobs, without resubmitting."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

REPO = Path(__file__).resolve().parents[2]
TARGETS = ("query_cartridges_prompt_aligned_dev_20261104", "query_cartridges_strong_value_gpt_dev_20261104")
CENTRAL = "query_cartridges_request_local_real_probe_20261104"


def verify_supporting_runtime(root):
    frozen = root / "runtime/repo/src/masked_ttt"
    source = REPO / "src/masked_ttt"
    if not frozen.is_dir():
        raise ValueError("The submitted job does not have its frozen runtime")
    for item in source.glob("*.py"):
        if item.name in ("pipeline.py", "cli.py"):
            continue
        if not (frozen / item.name).is_file() or (frozen / item.name).read_bytes() != item.read_bytes():
            raise ValueError("A supporting runtime module differs: " + item.name)


def wrapper(central):
    return ("from importlib.util import module_from_spec, spec_from_file_location\n"
            "from pathlib import Path\n"
            f"CENTRAL = Path({str(central)!r})\n"
            "def run_if_needed(engine, model, identity):\n"
            "    spec = spec_from_file_location('query_cartridges_shared_request_local_probe', CENTRAL / 'driver.py')\n"
            "    module = module_from_spec(spec)\n"
            "    spec.loader.exec_module(module)\n"
            "    return module.run_if_needed(engine, model, identity)\n")


def atomic_directory(path, files):
    """Publish complete marker and driver together; never overwrite other work."""
    if path.exists():
        if not path.is_dir() or any(not (path / name).is_file() or (path / name).read_bytes() != data
                                    for name, data in files.items()):
            raise ValueError("An existing optional probe has a different configuration")
        return
    stage = Path(tempfile.mkdtemp(prefix=".request-local-probe-", dir=path.parent))
    try:
        for name, data in files.items():
            (stage / name).write_bytes(data)
        os.rename(stage, path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def enable(workspace):
    workspace = Path(workspace).resolve()
    roots = [workspace / name for name in TARGETS]
    for root in roots:
        verify_supporting_runtime(root)
        driver = root / "runtime/rollout_driver.py"
        if not driver.is_file() or "disabled_optional_probe" not in driver.read_text():
            raise ValueError("This submitted job has no compatible optional hook")
    central = workspace / CENTRAL
    files = {
        "driver.py": (REPO / "experiments/kv_only_cartridge/in_job_probe.py").read_bytes(),
        "pipeline_candidate.py": (REPO / "src/masked_ttt/pipeline.py").read_bytes(),
        "ready.json": (json.dumps(dict(allow_single_loaded_model_probe=True, probe_version=2), indent=2) + "\n").encode(),
        "protocol.json": (json.dumps(dict(models=["qwen", "gpt"], synthetic_request=True,
            online_api_calls=0, reference_answers_used=False, new_gpu_jobs=0,
            training_runtime_changed=False, results_locked_per_model=True), indent=2) + "\n").encode(),
    }
    atomic_directory(central, files)
    optional = {
        "driver.py": wrapper(central).encode(),
        "ready.json": (json.dumps(dict(enabled=True, shared_probe=str(central)), indent=2) + "\n").encode(),
    }
    for root in roots:
        atomic_directory(root / "disabled_optional_probe", optional)
    return dict(enabled=True, roots=[str(root) for root in roots], shared_results=str(central),
                new_gpu_jobs=0, training_runtime_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(enable(args.workspace), indent=2))


if __name__ == "__main__":
    main()
