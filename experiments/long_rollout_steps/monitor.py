#!/usr/bin/env python3
"""Refresh a single pilot report hourly and stop the timer on completion."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root.resolve()
    if root.name != "query_cartridges_longsteps_pilot_20261104":
        raise ValueError("Unexpected pilot directory")
    subprocess.run([sys.executable, str(root / "report.py")], check=True, timeout=120)
    if (root / "finished.json").exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", "query-cartridges-longsteps-report.timer"],
                       check=True, capture_output=True, text=True)
        print("Pilot complete")
    else:
        print("Pilot report refreshed")


if __name__ == "__main__":
    main()
