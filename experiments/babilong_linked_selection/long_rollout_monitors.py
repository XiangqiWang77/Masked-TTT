"""Hourly local timers for the single combined study; never call the API."""
from pathlib import Path
import shlex
import subprocess

from experiments.babilong_linked_selection.prepare_long_rollout_job import REPO, atomic_install


def unit_contents(workspace, root, kind):
    if kind not in ("submit", "dev", "holdout"):
        raise ValueError("Unknown timer kind")
    if any(ch.isspace() for ch in str(workspace) + str(root)):
        raise ValueError("Systemd paths cannot contain whitespace")
    prefix = "query-cartridges-linked32-" + kind
    runtime_repo = root / "runtime/repo"
    script = (runtime_repo / "experiments/babilong_linked_selection/submit_long_rollout_job.py" if kind == "submit"
              else runtime_repo / "experiments/babilong_linked_selection/long_rollout_monitor.py")
    args = ["/usr/bin/python3", str(script), "--workspace", str(workspace), "--job-root", str(root)]
    if kind != "submit":
        args += ["--kind", kind]
    command = " ".join(shlex.quote(arg) for arg in args)
    service = ("[Unit]\nDescription=Update linked 32-token Query Cartridge " + kind + "\n\n"
        "[Service]\nType=oneshot\n"
        "Environment=PATH=/opt/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\n"
        "ExecStart=" + command + "\n")
    timer = ("[Unit]\nDescription=Hourly linked 32-token Query Cartridge " + kind + "\n\n"
        "[Timer]\nOnActiveSec=1h\nOnUnitActiveSec=1h\nUnit=" + prefix + ".service\n\n"
        "[Install]\nWantedBy=timers.target\n")
    return prefix, service, timer


def enable_timer(workspace, root, kind):
    prefix, service, timer = unit_contents(workspace, root, kind)
    directory = Path.home() / ".config/systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_install(directory / (prefix + ".service"), service)
    atomic_install(directory / (prefix + ".timer"), timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True,
                   capture_output=True, text=True, timeout=20)
    subprocess.run(["systemctl", "--user", "enable", "--now", prefix + ".timer"],
                   check=True, capture_output=True, text=True, timeout=20)


def disable_timer(kind):
    subprocess.run(["systemctl", "--user", "disable", "--now",
                    "query-cartridges-linked32-" + kind + ".timer"],
                   check=False, capture_output=True, text=True, timeout=20)
