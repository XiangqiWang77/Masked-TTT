#!/usr/bin/env python3
"""Run the frozen one-GPU original-versus-canonical evidence comparison."""
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
    driver = root / "runtime/rollout_driver.py"
    if not driver.is_file() or not (root / "manifest.json").is_file():
        raise ValueError("The frozen driver and manifest are required")
    spec = importlib.util.spec_from_file_location("frozen_canonical_pair_driver", driver)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    implementation.CROSS, implementation.FROZEN, implementation.REPO = root.parent, root / "runtime", REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(root / "runtime/helpers")]

    import masked_ttt as runtime
    import experiments.anchored_short_verifier.paired_controls as controls
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from experiments.value_lr_physical.canonical_runner import canonical_pair_runner
    from masked_ttt.linked_selection import LinkedFrozenSelector

    with canonical_pair_runner(implementation, runtime, controls, root=root, selector=LinkedFrozenSelector,
                               physical_controls=kv_only_original_question_controls):
        implementation.validate_manifest(json.loads((root / "manifest.json").read_text()))
        implementation.main()


if __name__ == "__main__":
    main()
