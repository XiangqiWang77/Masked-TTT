"""Attach aggregate KV accounting to existing hourly canonical job reports."""
from pathlib import Path
import re
import subprocess


def install(unit_dir, script, workspace, *, reload=True):
    unit_dir, script, workspace = Path(unit_dir), Path(script), Path(workspace)
    if any(not path.is_absolute() or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path))
           for path in (unit_dir, script, workspace)):
        raise ValueError("Use absolute, simple paths for systemd arguments")
    if not script.is_file():
        raise ValueError("The frozen memory-report script is required")
    for stage in ("dev", "holdout"):
        root = workspace / f"query_cartridges_canonical_pair_{stage}_20261104"
        dropin = unit_dir / f"query-cartridges-canonical-{stage}.service.d"
        dropin.mkdir(parents=True, exist_ok=True)
        path = dropin / "memory-report.conf"
        content = ("[Service]\n" + f"ExecStartPost=/usr/bin/python3 {script} "
                   f"--candidate-root {root} --output {root / 'memory_latest'}\n")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    if reload:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True, text=True)
