#!/usr/bin/env python3
"""Attach a lightweight physical-probe report to an existing hourly timer."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.counterfactual_transfer.enable_in_job_probe import CENTRAL, TARGET

SERVICE="query-cartridges-gain-rate-dev.service"


def _write_once(path,data):
    if path.exists():
        if not path.is_file() or path.read_bytes()!=data:
            raise ValueError("An existing report file has different content")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=".report-",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as file:
            file.write(data)
        # A hard link atomically publishes without overwriting another file.
        os.link(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def install(workspace,root,*,unit_root=None,reload=True):
    workspace,root=Path(workspace).resolve(),Path(root).resolve()
    if root.parent!=workspace or root.name!=TARGET or not (root/"submission.json").is_file():
        raise ValueError("Use the existing gain-rate development job")
    central=workspace/CENTRAL
    protocol=json.loads((central/"protocol.json").read_text())
    ready=json.loads((root/"disabled_optional_probe/ready.json").read_text())
    if (not protocol.get("scores_after_full_and_source_release") or
        protocol.get("extra_gpu_jobs")!=0 or ready.get("shared_probe")!=str(central)):
        raise ValueError("The physical counterfactual probe is not installed")
    units=(Path(os.environ.get("XDG_CONFIG_HOME",str(Path.home()/".config")))/"systemd/user"
           if unit_root is None else Path(unit_root))
    service=units/SERVICE
    if not service.is_file() or str(root) not in service.read_text():
        raise ValueError("The existing hourly monitor does not target this job")
    reporter=central/"report.py"
    _write_once(reporter,(REPO/"experiments/counterfactual_transfer/report_in_job.py").read_bytes())
    command=f"/usr/bin/python3 {shlex.quote(str(reporter))} --results-root {shlex.quote(str(central))}"
    dropin=units/(SERVICE+".d")/"physical-counterfactual.conf"
    _write_once(dropin,("[Service]\nExecStartPost="+command+"\n").encode())
    if reload:
        subprocess.run(["systemctl","--user","daemon-reload"],check=True,capture_output=True,text=True)
    return dict(installed=True,reporter=str(reporter),dropin=str(dropin),extra_gpu_jobs=0,
                reporting_interval="existing hourly timer")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",required=True,type=Path)
    parser.add_argument("--job-root",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(install(args.workspace,args.job_root),indent=2))


if __name__=="__main__":main()
