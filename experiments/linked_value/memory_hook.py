"""Attach GPT linked-value reports to the existing hourly service."""
from pathlib import Path
import re
import subprocess


def install(unit_dir, script, workspace, *, kind="memory", reload=True):
    unit_dir, script, workspace = Path(unit_dir), Path(script), Path(workspace)
    if any(not path.is_absolute() or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path))
           for path in (unit_dir, script, workspace)):
        raise ValueError("Use absolute, simple paths for systemd arguments")
    if not script.is_file():
        raise ValueError("The frozen report script is required")
    if kind not in ("memory", "detailed"):
        raise ValueError("Unknown report kind")
    root = workspace / "query_cartridges_linked_value_gpt_holdout_20261104"
    dropin = unit_dir / "query-cartridges-linked-value-holdout.service.d"
    dropin.mkdir(parents=True, exist_ok=True)
    path = dropin / f"{kind}-report.conf"
    content = ("[Service]\n" + f"ExecStartPost=/usr/bin/python3 {script} "
               f"--job-root {root} --output {root / (kind + '_latest')}\n")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content)
    temporary.replace(path)
    if reload:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
