#!/usr/bin/env python3
"""Run the existing one-GPU 32-token benchmark with linked 10% evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path, help="Existing 32-token runner")
    parser.add_argument("--job-root", required=True, type=Path, help="Directory with frozen manifest and report.py")
    args = parser.parse_args()
    job_root, driver = args.job_root.resolve(), args.driver.resolve()
    if not (job_root / "manifest.json").is_file() or not (job_root / "report.py").is_file():
        raise ValueError("A frozen manifest and report are required before execution")
    spec = importlib.util.spec_from_file_location("original_rollout_runner", driver)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the original 32-token runner")
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    if driver != job_root / "runtime/rollout_driver.py":
        raise ValueError("The runner must use this job's frozen rollout driver")
    implementation.CROSS = job_root.parent
    implementation.FROZEN = job_root / "runtime"
    implementation.REPO = REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(job_root / "runtime/helpers")]

    import masked_ttt as runtime
    import experiments.anchored_short_verifier.paired_controls as paired_controls
    from experiments.babilong_linked_selection.long_rollout_runtime import linked_long_rollout_runner
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from masked_ttt.linked_selection import LinkedFrozenSelector

    with linked_long_rollout_runner(implementation, runtime, paired_controls, root=job_root,
                                   linked_selector=LinkedFrozenSelector,
                                   physical_controls=kv_only_original_question_controls):
        implementation.validate_manifest(json.loads((job_root / "manifest.json").read_text()))
        implementation.main()


if __name__ == "__main__":
    main()
