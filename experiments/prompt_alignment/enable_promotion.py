#!/usr/bin/env python3
"""Freeze the post-development promotion controller and enable an hourly timer."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.gqa_head_gain.prepare import atomic

TIMER = "query-cartridges-prompt-aligned-promotion.timer"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "query_cartridges_prompt_aligned_control_20261104"
    frozen = root / "runtime/repo"
    if (root / "ready.json").is_file():
        if not (frozen / "experiments/prompt_alignment/promote.py").is_file():
            raise ValueError("The existing promotion controller is incomplete")
    else:
        if frozen.exists():
            raise ValueError("An unfinished controller needs inspection")
        frozen.mkdir(parents=True)
        ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
        for folder in ("src", "experiments", "tests", "examples"):
            shutil.copytree(REPO / folder, frozen / folder, ignore=ignore)
        env = dict(os.environ, PYTHONPATH="src:tests:.", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        check = subprocess.run([str(args.python), "-m", "pytest", "--rootdir", str(frozen),
            "--confcutdir", str(frozen), "-q", "tests/test_prompt_alignment_holdout.py"], cwd=frozen,
            env=env, capture_output=True, text=True, timeout=120)
        if check.returncode or "3 passed" not in check.stdout:
            raise RuntimeError("Frozen promotion controller validation failed:\n" + check.stdout + check.stderr)
        atomic(root / "ready.json", json.dumps(dict(cpu_tests_passed=True,
            gpu_preflight_required=True, online_api_calls=0, created=time.time()), indent=2) + "\n")
    unit = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    service = ("[Unit]\nDescription=Promote matched aligned-question SD when eligible\n\n"
        "[Service]\nType=oneshot\n"
        "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
        f"ExecStart=/usr/bin/python3 {frozen}/experiments/prompt_alignment/promote.py "
        f"--workspace {workspace} --control-root {root} --python {args.python} --timer {TIMER}\n")
    timer = ("[Unit]\nDescription=Check aligned-question SD promotion once per hour\n\n"
        "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nAccuracySec=1m\n\n"
        "[Install]\nWantedBy=timers.target\n")
    atomic(unit / "query-cartridges-prompt-aligned-promotion.service", service)
    atomic(unit / TIMER, timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", TIMER], check=True, capture_output=True, text=True)
    print(json.dumps(dict(enabled=True, timer=TIMER, online_api_calls=0)))


if __name__ == "__main__":
    main()
