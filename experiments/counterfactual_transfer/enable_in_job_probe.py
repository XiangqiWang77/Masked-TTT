#!/usr/bin/env python3
"""Install a read-only probe in an existing job's optional preflight hook."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

REPO = Path(__file__).resolve().parents[2]
TARGET = "query_cartridges_gain_rate_pair_dev_20261104"
CENTRAL = "query_cartridges_physical_counterfactual_probe_20261104"


def verify_supporting_runtime(root):
    source, frozen = REPO / "src/masked_ttt", root / "runtime/repo/src/masked_ttt"
    if not (root / "submission.json").is_file() or not frozen.is_dir():
        raise ValueError("An already submitted frozen job is required")
    for item in source.glob("*.py"):
        if not (frozen / item.name).is_file() or (frozen / item.name).read_bytes() != item.read_bytes():
            raise ValueError("A supporting runtime module differs: " + item.name)
    driver = root / "runtime/rollout_driver.py"
    if not driver.is_file() or "disabled_optional_probe" not in driver.read_text():
        raise ValueError("This job does not have a compatible optional hook")


def _wrapper(central):
    return ("from importlib.util import module_from_spec, spec_from_file_location\n"
            "from pathlib import Path\n"
            f"CENTRAL = Path({str(central)!r})\n"
            "def run_if_needed(engine, model, identity):\n"
            "    spec = spec_from_file_location('query_cartridges_physical_probe_driver', CENTRAL / 'driver.py')\n"
            "    module = module_from_spec(spec)\n"
            "    spec.loader.exec_module(module)\n"
            "    return module.run_if_needed(engine, model, identity)\n")


def _publish_directory(path, files):
    """Publish marker and complete code together without replacing other work."""
    if path.exists():
        if (not path.is_dir() or any(not (path / name).is_file() or (path / name).read_bytes() != data
                                     for name,data in files.items())):
            raise ValueError("An existing optional probe has a different configuration")
        return
    stage = Path(tempfile.mkdtemp(prefix=".physical-probe-", dir=path.parent))
    try:
        for name,data in files.items():
            (stage / name).write_bytes(data)
        os.rename(stage,path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def enable(workspace, root):
    workspace, root = Path(workspace).resolve(), Path(root).resolve()
    if root.parent != workspace or root.name != TARGET:
        raise ValueError("Use the submitted gain-rate paired development job")
    verify_supporting_runtime(root)
    central = workspace / CENTRAL
    files = {
        "driver.py": (REPO / "experiments/counterfactual_transfer/in_job_probe.py").read_bytes(),
        "probe.py": (REPO / "experiments/counterfactual_transfer/probe.py").read_bytes(),
        "ready.json": (json.dumps(dict(enabled=True, version="physical-kv-v1"), indent=2) + "\n").encode(),
        "protocol.json": (json.dumps(dict(models=["qwen","gpt"], synthetic_request=True,
            reference_answers_used=False, online_api_calls=0, extra_gpu_jobs=0,
            training_runtime_changed=False, scores_after_full_and_source_release=True), indent=2) + "\n").encode(),
    }
    _publish_directory(central,files)
    option = {"driver.py":_wrapper(central).encode(),
              "ready.json":(json.dumps(dict(enabled=True, shared_probe=str(central)),indent=2)+"\n").encode()}
    _publish_directory(root / "disabled_optional_probe",option)
    return dict(enabled=True, root=str(root), shared_results=str(central), extra_gpu_jobs=0,
                training_runtime_changed=False, runs_only_if_allocation_reaches_optional_hook=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",required=True,type=Path)
    parser.add_argument("--job-root",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(enable(args.workspace,args.job_root),indent=2))


if __name__ == "__main__":
    main()
