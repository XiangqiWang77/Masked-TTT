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
OLD_CENTRAL = "query_cartridges_request_local_real_probe_20261104"
CENTRAL = "query_cartridges_request_local_real_probe_v3_20261104"


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
    for name in ("experiments/kv_only_cartridge/driver.py",
                 "experiments/kv_only_cartridge/compact.py",
                 "experiments/shared_question_sd/controls.py"):
        staged, current = root / "runtime/repo" / name, REPO / name
        if not staged.is_file() or staged.read_bytes() != current.read_bytes():
            raise ValueError("A supporting runtime module differs: " + name)


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


def central_files():
    return {
        "driver.py": (REPO / "experiments/kv_only_cartridge/in_job_probe.py").read_bytes(),
        "pipeline_candidate.py": (REPO / "src/masked_ttt/pipeline.py").read_bytes(),
        "paired_control_candidate.py": (REPO / "experiments/kv_only_cartridge/paired_control.py").read_bytes(),
        "ready.json": (json.dumps(dict(allow_single_loaded_model_probe=True, probe_version=3), indent=2) + "\n").encode(),
        "protocol.json": (json.dumps(dict(models=["qwen", "gpt"], synthetic_request=True,
            online_api_calls=0, reference_answers_used=False, new_gpu_jobs=0,
            training_runtime_changed=False, results_locked_per_model=True), indent=2) + "\n").encode(),
    }


def validated_roots(workspace):
    roots = [workspace / name for name in TARGETS]
    for root in roots:
        verify_supporting_runtime(root)
        driver = root / "runtime/rollout_driver.py"
        if not driver.is_file() or "disabled_optional_probe" not in driver.read_text():
            raise ValueError("This submitted job has no compatible optional hook")
    return roots


def enable(workspace):
    workspace = Path(workspace).resolve()
    roots = validated_roots(workspace)
    central = workspace / CENTRAL
    files = central_files()
    atomic_directory(central, files)
    optional = {
        "driver.py": wrapper(central).encode(),
        "ready.json": (json.dumps(dict(enabled=True, shared_probe=str(central)), indent=2) + "\n").encode(),
    }
    for root in roots:
        atomic_directory(root / "disabled_optional_probe", optional)
    return dict(enabled=True, roots=[str(root) for root in roots], shared_results=str(central),
                new_gpu_jobs=0, training_runtime_changed=False)


def upgrade_existing(workspace):
    """Keep any in-flight v2 probe separate while routing future hooks to v3."""
    workspace = Path(workspace).resolve()
    roots = validated_roots(workspace)
    old, central = workspace / OLD_CENTRAL, workspace / CENTRAL
    readiness = json.loads((old / "ready.json").read_text())
    if readiness.get("probe_version") != 2:
        raise ValueError("Expected the existing version 2 probe")
    old_driver, new_driver = wrapper(old).encode(), wrapper(central).encode()
    for root in roots:
        option = root / "disabled_optional_probe"
        if not (option / "ready.json").is_file() or (option / "driver.py").read_bytes() not in (old_driver, new_driver):
            raise ValueError("An existing optional probe has a different configuration")
    atomic_directory(central, central_files())
    for root in roots:
        option = root / "disabled_optional_probe"
        for name, data in (("driver.py", new_driver), ("ready.json", (json.dumps(
                dict(enabled=True, shared_probe=str(central)), indent=2) + "\n").encode())):
            temporary = option / ("." + name + ".v3.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, option / name)
    return dict(enabled=True, roots=[str(root) for root in roots], shared_results=str(central),
                previous_results_preserved=str(old), new_gpu_jobs=0, training_runtime_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--upgrade-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps((upgrade_existing if args.upgrade_existing else enable)(args.workspace), indent=2))


if __name__ == "__main__":
    main()
