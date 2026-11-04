#!/usr/bin/env python3
"""Execute the frozen one-GPU linked-evidence/query-head comparison."""
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
    args = parser.parse_args()
    root = args.job_root.resolve()
    driver = root / "runtime/rollout_driver.py"
    if not driver.is_file() or not (root / "manifest.json").is_file():
        raise ValueError("The frozen driver and manifest are required")
    spec = importlib.util.spec_from_file_location("frozen_olmo_style_runner", driver)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    implementation.CROSS = root.parent
    implementation.FROZEN = root / "runtime"
    implementation.REPO = REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(root / "runtime/helpers")]

    import masked_ttt as runtime
    import experiments.anchored_short_verifier.paired_controls as paired_controls
    from experiments.gqa_head_gain.runner import query_head_runner
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from masked_ttt.linked_selection import LinkedFrozenSelector

    with query_head_runner(implementation, runtime, paired_controls, root=root,
                           linked_selector=LinkedFrozenSelector,
                           physical_controls=kv_only_original_question_controls):
        implementation.validate_manifest(json.loads((root / "manifest.json").read_text()))
        implementation.main()


if __name__ == "__main__":
    main()
