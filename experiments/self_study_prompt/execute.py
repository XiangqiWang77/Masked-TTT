#!/usr/bin/env python3
"""Execute the frozen, one-GPU, same-request study-prompt comparison."""
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
    path = root / "runtime/rollout_driver.py"
    if not path.is_file() or not (root / "manifest.json").is_file():
        raise ValueError("The frozen driver and manifest are required")
    spec = importlib.util.spec_from_file_location("frozen_study_pair_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.CROSS, driver.FROZEN, driver.REPO = root.parent, root / "runtime", REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(root / "runtime/helpers")]
    import masked_ttt as runtime
    import experiments.anchored_short_verifier.paired_controls as controls
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from experiments.self_study_prompt.study_runner import study_pair_runner
    from masked_ttt.linked_selection import LinkedFrozenSelector

    with study_pair_runner(driver, runtime, controls, root=root, selector=LinkedFrozenSelector,
                           physical_controls=kv_only_original_question_controls):
        driver.main()


if __name__ == "__main__":
    main()
