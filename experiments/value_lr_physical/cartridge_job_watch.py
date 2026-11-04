"""Read-only liveness checks for hourly Query Cartridge reports."""
import json
from pathlib import Path
import re
import subprocess
import time

TERMINAL = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "COMPLETED",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
REQUEUE_TRANSIENT = {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "FAILED", "OUT_OF_MEMORY", "TIMEOUT"}


def _query(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None


def observed_terminal(job):
    """Return a terminal state only when scheduler evidence supports it.

    A temporarily absent queue entry, failed query, or accounting row from a
    previous allocation is not enough to conclude that a requeued job stopped.
    """
    job = str(job)
    if not job.isdigit():
        raise ValueError("Invalid Slurm job id")
    queue = _query(["squeue", "-h", "-j", job, "-o", "%i"])
    if queue is None or queue.returncode or job in queue.stdout.split():
        return None
    control = _query(["scontrol", "show", "job", job, "-o"])
    if control is None:
        return None
    if control.returncode == 0:
        match = re.search(r"\bJobState=([A-Z_]+)", control.stdout)
        if not match:
            return None
        state = match.group(1)
        requeue = re.search(r"\bRequeue=(\d+)", control.stdout)
        if state in REQUEUE_TRANSIENT and requeue and int(requeue.group(1)):
            return None
        return state if state in TERMINAL else None
    if "Invalid job id" not in getattr(control, "stderr", ""):
        return None
    accounting = _query(["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State%32"])
    if accounting is None or accounting.returncode:
        return None
    states = []
    for line in accounting.stdout.splitlines():
        fields = line.split("|")
        if len(fields) > 1 and fields[0] == job:
            words = fields[1].split()
            if words:
                states.append(words[0].split("+")[0])
    if not states or len(set(states)) != 1 or states[0] not in TERMINAL:
        return None
    # Recheck after accounting: the job can reappear while Slurm requeues it.
    queue = _query(["squeue", "-h", "-j", job, "-o", "%i"])
    if queue is None or queue.returncode or job in queue.stdout.split():
        return None
    return states[0]


def record_terminal(root):
    root = Path(root)
    submission = root / "submission.json"
    if not submission.exists():
        return None
    job = json.loads(submission.read_text())["job_id"]
    state = observed_terminal(job)
    if state is None:
        return None
    target = root / "terminal_job_state.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(job_id=job, state=state, observed=time.time())) + "\n")
    temporary.replace(target)
    return state
