#!/usr/bin/env python3
"""Refresh a physical 8-step holdout hourly and combine on completion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from experiments.gqa_head_gain.prepare import atomic
from experiments.step_physical.report import combine, write
from experiments.value_lr_physical.recovery_report import collect as recovery_collect, write as recovery_write


def advance(workspace, root):
    workspace, root = workspace.resolve(), root.resolve()
    if root.parent != workspace or root.name != "query_cartridges_step_physical_gpt_holdout_20261104":
        raise ValueError("Unexpected physical 8-step holdout directory")
    subprocess.run([sys.executable, str(root / "report.py")], check=True,
                   capture_output=True, text=True, timeout=120)
    holdout = json.loads((root / "latest.json").read_text())
    if (root / "finished.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        development_manifest = json.loads((root / "development_manifest.json").read_text())
        development = json.loads((root / "development_snapshot.json").read_text())
        combined = combine(development_manifest, development, manifest, holdout)
        if combined is None:
            raise ValueError("The completed holdout does not have 800 valid scored cases")
        write(combined, root / "combined_1000")
        recovery_write(recovery_collect(combined), root / "recovery_latest")
        return dict(stage="complete", report=str(root / "combined_1000.md"))
    recovery_write(recovery_collect(holdout), root / "recovery_latest")
    sys.path.insert(0, str(workspace))
    from cartridge_job_watch import record_terminal
    terminal = record_terminal(root)
    return dict(stage="terminal_without_complete_results" if terminal else "in_progress",
                scheduler_state=terminal, cases=holdout["overall"]["n"])


def enable_monitor(workspace, root):
    name = "query-cartridges-step-physical-gpt-report"
    status = subprocess.run(["systemctl", "--user", "is-active", name + ".timer"],
                            capture_output=True, text=True, timeout=20, check=False)
    if status.returncode == 0 and status.stdout.strip() == "active":
        return name + ".timer"
    script = root / "runtime/repo/experiments/step_physical/monitor.py"
    command = ["systemd-run", "--user", "--unit=" + name, "--on-active=1h", "--on-unit-active=1h",
        "--timer-property=AccuracySec=1m", "--setenv=PATH=/opt/slurm/current/bin:/usr/local/bin:/usr/bin:/bin",
        "--setenv=OPENAI_API_KEY=", "--setenv=OPENAI_API_KEY_FILE=", "/usr/bin/python3", str(script),
        "--workspace", str(workspace), "--job-root", str(root), "--timer", name + ".timer"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return name + ".timer"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--timer", required=True)
    args = parser.parse_args()
    root = args.job_root.resolve()
    try:
        result = advance(args.workspace, root)
    except Exception as exc:
        atomic(root / "monitor_status.json", json.dumps(
            dict(stage="monitor_error", error_type=type(exc).__name__, updated=time.time()), indent=2) + "\n")
        raise
    atomic(root / "monitor_status.json", json.dumps(dict(result, updated=time.time()), indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["stage"] in ("complete", "terminal_without_complete_results"):
        subprocess.run(["systemctl", "--user", "stop", args.timer],
                       check=True, capture_output=True, text=True, timeout=20)


if __name__ == "__main__":
    main()
