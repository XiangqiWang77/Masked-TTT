#!/usr/bin/env python3
"""Run one source-disjoint, development-fixed Value LR on one GPU."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    root = parser.parse_args().job_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    driver = root / "runtime/rollout_driver.py"
    if not driver.is_file():
        raise ValueError("The frozen original-SD runner is required")
    spec = importlib.util.spec_from_file_location("frozen_value_holdout", driver)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    implementation.CROSS = root.parent
    implementation.FROZEN = root / "runtime"
    implementation.REPO = REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(root / "runtime/helpers")]
    import experiments.anchored_short_verifier.paired_controls as paired_controls
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from experiments.value_lr_physical.runner import physical_value_runner

    with physical_value_runner(implementation, paired_controls, root=root,
            fixed_lr=manifest["fixed_value_learning_rate"],
            physical_controls=kv_only_original_question_controls):
        implementation.validate_manifest(manifest)
        implementation.main()


if __name__ == "__main__":
    main()
