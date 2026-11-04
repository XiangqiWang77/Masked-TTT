"""Source-only study-prompt gate executed on the already allocated GPU."""
import importlib.util
import json
import os
from pathlib import Path
import socket

from experiments.gqa_head_gain.prepare import atomic
from experiments.self_study_prompt.report import summarize
from experiments.self_study_prompt.study_gate import probe_decision


class ProbeRejected(Exception):
    def __init__(self, decision):
        super().__init__("The source-only study probe did not improve verified exercises")
        self.decision = decision


class ProbeDeferred(Exception):
    def __init__(self, progress):
        super().__init__("The source-only study probe needs another allocation")
        self.progress = progress


class ProbeFailed(Exception):
    def __init__(self, progress):
        super().__init__("The source-only study probe recorded an execution error")
        self.progress = progress


def run_inline_probe(engine, root):
    """Run at most ten label-free probe requests before any development case."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["models"] != ["qwen"] or manifest["probe_gate_mode"] != "in_job_before_dataset" or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The inline study-probe protocol changed")
    if manifest["mode"] == "holdout":
        summary = json.loads((root / "probe_verified.json").read_text())
    elif manifest["mode"] == "development":
        import torch
        directory = root / "study_probe"
        spec = importlib.util.spec_from_file_location("request_local_study_probe", directory / "driver.py")
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        identity = dict(job=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(engine.device),
                        node=os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
                        allocation_restart_count=os.environ.get("SLURM_RESTART_COUNT", "0"))
        driver.run_if_needed(engine, "qwen", identity)
        path = directory / "qwen.json"
        if not path.is_file():
            raise ProbeDeferred(dict(n=0, expected=10, errors=0))
        summary = summarize(json.loads(path.read_text()))
        atomic(directory / "latest.json", json.dumps(summary, indent=2) + "\n")
        if summary["state"] != "complete":
            progress = dict(n=summary["n"], expected=summary["expected"], errors=summary["errors"])
            if summary["errors"]:
                raise ProbeFailed(progress)
            raise ProbeDeferred(progress)
        atomic(root / "probe_verified.json", json.dumps(summary, indent=2) + "\n")
    else:
        raise ValueError("Unknown study-probe experiment mode")
    decision = probe_decision(summary)
    if not decision["eligible"]:
        raise ProbeRejected(decision)
    return dict(source_probe_complete=True, source_probe_more_verified=True,
                source_probe_reference_answers_used=False, source_probe_online_api_calls=0)
