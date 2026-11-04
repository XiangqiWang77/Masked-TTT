#!/usr/bin/env python3
"""Build an isolated offline wheel and exercise both public cartridge paths."""
import argparse
import configparser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="A new scratch directory; existing data is never overwritten")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    stage, wheels = work / "source", work / "wheels"
    stage.mkdir()
    wheels.mkdir()
    shutil.copytree(root / "src", stage / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(root / name, stage / name)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    build = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                            "--no-index", "--wheel-dir", str(wheels), "."],
                           cwd=stage, env=env, capture_output=True, text=True, timeout=180)
    if build.returncode:
        raise RuntimeError("The isolated wheel build failed:\n" + build.stderr)
    candidates = list(wheels.glob("*.whl"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one built wheel")
    wheel = candidates[0]
    with zipfile.ZipFile(wheel) as archive:
        entries = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
        if len(entries) != 1:
            raise ValueError("The wheel has no unique CLI entry point")
        config = configparser.ConfigParser()
        config.read_string(archive.read(entries[0]).decode())
        if config["console_scripts"]["masked-ttt"] != "masked_ttt.cli:main":
            raise ValueError("The wheel CLI entry point changed")
    env["PYTHONPATH"] = os.pathsep.join((str(wheel), str(root / "tests")))
    help_result = subprocess.run([sys.executable, "-m", "masked_ttt.cli", "--help"], cwd=work,
                                 env=env, capture_output=True, text=True, timeout=60)
    if (help_result.returncode or "--score-json-dir" not in help_result.stdout or
        "--selection-policy" not in help_result.stdout):
        raise RuntimeError("The packaged CLI failed its offline help check")
    smoke = subprocess.run([sys.executable, str(root / "tests/wheel_smoke.py")], cwd=work,
                           env=env, capture_output=True, text=True, timeout=180)
    if smoke.returncode:
        raise RuntimeError("The packaged cartridge smoke test failed:\n" + smoke.stdout + smoke.stderr)
    print("Isolated wheel and CLI entry point built offline")
    print(smoke.stdout, end="")


if __name__ == "__main__":
    main()
