#!/usr/bin/env python3
"""Run a frozen one-GPU BF16/FP32 cartridge materialization comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.job_root.resolve()
    driver = root / "runtime/rollout_driver.py"
    if not driver.is_file() or not (root / "manifest.json").is_file():
        raise ValueError("The frozen driver and manifest are required")
    spec = importlib.util.spec_from_file_location("frozen_precision_runner", driver)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    implementation.CROSS = root.parent
    implementation.FROZEN = root / "runtime"
    implementation.REPO = REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(root / "runtime/helpers")]

    import masked_ttt as runtime
    import experiments.anchored_short_verifier.paired_controls as paired_controls
    from experiments.precision_cartridge.benchmark_runner import precision_runner
    from masked_ttt.linked_selection import LinkedFrozenSelector

    with precision_runner(implementation, runtime, paired_controls, root=root,
                          linked_selector=LinkedFrozenSelector):
        implementation.validate_manifest(json.loads((root / "manifest.json").read_text()))
        implementation.main()


if __name__ == "__main__":
    main()
