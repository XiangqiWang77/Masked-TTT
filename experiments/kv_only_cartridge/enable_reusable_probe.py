#!/usr/bin/env python3
"""Attach a synthetic reusable-KV preflight to existing optional GPU hooks."""
import argparse
import json
from pathlib import Path

from experiments.kv_only_cartridge.enable_in_job_probe import atomic_directory

REPO = Path(__file__).resolve().parents[2]
TARGETS = ("query_cartridges_rollout_value_qwen_dev_20261104",
           "query_cartridges_linked_value_gpt_holdout_20261104")
CENTRAL = "query_cartridges_reusable_gpu_probe_20261104"
SUPPORT = ("cache.py", "memory_accounting.py", "runtime.py", "standalone.py", "kv_only.py",
           "attention.py", "reader.py", "architectures.py", "model.py")


def wrapper(central):
    return ("from importlib.util import module_from_spec, spec_from_file_location\n"
            "from pathlib import Path\n"
            f"CENTRAL = Path({str(central)!r})\n"
            "def run_if_needed(engine, model, identity):\n"
            "    spec = spec_from_file_location('query_cartridges_reusable_gpu_probe', CENTRAL / 'driver.py')\n"
            "    module = module_from_spec(spec)\n"
            "    spec.loader.exec_module(module)\n"
            "    return module.run_if_needed(engine, model, identity)\n")


def enable(workspace):
    workspace = Path(workspace).resolve()
    roots = [workspace / name for name in TARGETS]
    for root in roots:
        driver = root / "runtime/rollout_driver.py"
        frozen = root / "runtime/repo"
        if not driver.is_file() or "disabled_optional_probe" not in driver.read_text():
            raise ValueError("A submitted job has no compatible optional probe hook")
        for name in SUPPORT:
            current = REPO / "src/masked_ttt" / name
            staged = frozen / "src/masked_ttt" / name
            if not staged.is_file() or staged.read_bytes() != current.read_bytes():
                raise ValueError("A required frozen supporting module differs: " + name)
        relative = Path("experiments/kv_only_cartridge/in_job_probe.py")
        if not (frozen / relative).is_file() or (frozen / relative).read_bytes() != (REPO / relative).read_bytes():
            raise ValueError("The frozen synthetic probe helpers differ")
    central = workspace / CENTRAL
    files = {
        "driver.py": (REPO / "experiments/kv_only_cartridge/reusable_probe.py").read_bytes(),
        "pipeline_candidate.py": (REPO / "src/masked_ttt/pipeline.py").read_bytes(),
        "cartridge_candidate.py": (REPO / "src/masked_ttt/cartridge.py").read_bytes(),
        "ready.json": (json.dumps(dict(probe="reusable_selected_kv", version=1), indent=2) + "\n").encode(),
        "protocol.json": (json.dumps(dict(models=["qwen", "gpt"], synthetic_request=True,
            online_api_calls=0, reference_answers_used=False, new_gpu_jobs=0,
            training_runtime_changed=False, exact_same_question=True), indent=2) + "\n").encode(),
    }
    option = {"driver.py": wrapper(central).encode(),
              "ready.json": (json.dumps(dict(enabled=True, shared_probe=str(central)), indent=2) + "\n").encode()}
    # Check every destination before writing any, so an unrelated optional
    # hook is never replaced and cannot leave only one job newly modified.
    for path, expected in [(central, files)] + [(root / "disabled_optional_probe", option) for root in roots]:
        if path.exists() and (not path.is_dir() or any(not (path / name).is_file() or
            (path / name).read_bytes() != content for name, content in expected.items())):
            raise ValueError("An existing optional probe has a different configuration")
    atomic_directory(central, files)
    for root in roots:
        atomic_directory(root / "disabled_optional_probe", option)
    return dict(enabled=True, roots=[str(root) for root in roots], shared_results=str(central),
                new_gpu_jobs=0, training_runtime_changed=False,
                runs_only_when_existing_allocation_reaches_hook=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    print(json.dumps(enable(parser.parse_args().workspace), indent=2))


if __name__ == "__main__":
    main()
