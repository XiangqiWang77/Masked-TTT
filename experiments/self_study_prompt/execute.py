#!/usr/bin/env python3
"""Execute the frozen, one-GPU, same-request study-prompt comparison."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def deferred_probe_action(root, progress, *, now=None):
    """Bound retries and charge already-consumed GPU allocation time."""
    now = time.time() if now is None else now
    root = Path(root)
    old_path = root / "probe_deferred.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else None
    previous = old.get("progress", {}).get("n", -1) if old else -1
    stalls = (old.get("stalls", 0) + 1 if old and progress["n"] <= previous else 0)
    budget = json.loads((root / "allocation_budget.json").read_text())
    active = budget.get("active")
    spent = float(budget["spent_seconds"])
    if active:
        spent += min(float(active["remaining_seconds"]), max(0., now-float(active["started_wall"])))
    remaining = max(0., float(budget["total_budget_seconds"]) - spent)
    return dict(requeue=stalls < 2 and remaining > 900, stalls=stalls,
                remaining_budget_seconds=remaining, progress=progress)


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
    from experiments.gqa_head_gain.prepare import atomic
    from experiments.self_study_prompt.inline_probe import ProbeDeferred, ProbeFailed, ProbeRejected
    from experiments.self_study_prompt.study_runner import study_pair_runner
    from masked_ttt.linked_selection import LinkedFrozenSelector

    with study_pair_runner(driver, runtime, controls, root=root, selector=LinkedFrozenSelector,
                           physical_controls=kv_only_original_question_controls):
        try:
            driver.main()
        except ProbeRejected as exc:
            atomic(root / "probe_rejected.json", json.dumps(dict(decision=exc.decision), indent=2) + "\n")
        except ProbeFailed as exc:
            atomic(root / "probe_failed.json", json.dumps(dict(progress=exc.progress), indent=2) + "\n")
        except ProbeDeferred as exc:
            action = deferred_probe_action(root, exc.progress)
            atomic(root / "probe_deferred.json", json.dumps(action, indent=2) + "\n")
            if action["requeue"]:
                job = os.environ.get("SLURM_JOB_ID", "")
                if not job.isdigit():
                    raise ValueError("Cannot requeue a probe without its Slurm job ID")
                subprocess.run(["scontrol", "requeue", job], check=True, timeout=30)
            else:
                atomic(root / "probe_failed.json", json.dumps(dict(reason="probe_stalled_or_budget_exhausted",
                                                                    **action), indent=2) + "\n")


if __name__ == "__main__":
    main()
