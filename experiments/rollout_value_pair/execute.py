#!/usr/bin/env python3
"""Run the frozen Qwen same-GPU rollout/Value pair on cached Decision scores."""
import argparse
import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    root = parser.parse_args().job_root.resolve()
    spec = importlib.util.spec_from_file_location("frozen_rollout_value_driver", root / "runtime/rollout_driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.CROSS, driver.FROZEN, driver.REPO = root.parent, root / "runtime", REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(root / "runtime/helpers")]
    import masked_ttt as runtime
    import experiments.anchored_short_verifier.paired_controls as controls
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from experiments.rollout_value_pair.runner import pair_runner
    from masked_ttt.linked_selection import LinkedFrozenSelector

    with pair_runner(driver, runtime, controls, root=root, selector=LinkedFrozenSelector,
                     physical_controls=kv_only_original_question_controls):
        driver.main()


if __name__ == "__main__":
    main()
