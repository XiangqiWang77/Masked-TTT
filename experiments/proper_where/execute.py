#!/usr/bin/env python3
"""Execute the frozen one-GPU Qwen question-overlap study."""
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
    spec = importlib.util.spec_from_file_location("frozen_proper_where_runner", driver)
    implementation = importlib.util.module_from_spec(spec); spec.loader.exec_module(implementation)
    implementation.CROSS, implementation.FROZEN, implementation.REPO = root.parent, root / "runtime", REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(root / "runtime/helpers")]
    import masked_ttt as runtime
    import experiments.anchored_short_verifier.paired_controls as controls
    from experiments.babilong_linked_selection.proper_where_selector import ProperWhereSelector
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from experiments.proper_where.runner import proper_where_runner

    with proper_where_runner(implementation, runtime, controls, root=root, selector=ProperWhereSelector,
                             physical_controls=kv_only_original_question_controls):
        implementation.validate_manifest(json.loads((root / "manifest.json").read_text()))
        implementation.main()


if __name__ == "__main__":
    main()
