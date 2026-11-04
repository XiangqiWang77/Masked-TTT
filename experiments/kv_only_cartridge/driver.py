"""Run the optional KV-only preflight inside an already-allocated model job."""
import fcntl
import gc
import json
from pathlib import Path
import time
import traceback


def run_if_enabled(engine, model, identity, directory, *, probe=None):
    directory = Path(directory)
    marker = directory / "ready.json"
    if not marker.exists():
        return "not_enabled"
    if model not in ("qwen", "gpt"):
        raise ValueError("Unexpected model")
    readiness = json.loads(marker.read_text())
    if not readiness.get("allow_single_loaded_model_probe"):
        raise ValueError("Probe readiness marker is invalid")
    version = readiness.get("probe_version", 1)
    if type(version) is not int or version < 1:
        raise ValueError("Probe version must be a positive integer")

    import torch

    output = directory / (model + ".json")
    with (directory / (model + ".lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "already_running"
        previous = json.loads(output.read_text()) if output.exists() else None
        same_version = previous and previous.get("probe_version", 1) == version
        if same_version and previous.get("passed"):
            return "already_passed"
        if same_version and previous.get("attempts", 0) >= 2:
            return "previous_attempts_failed"
        if previous and not same_version:
            archive = directory / (model + ".v" + str(previous.get("probe_version", 1)) + ".json")
            if not archive.exists():
                archive.write_text(json.dumps(previous, indent=2, ensure_ascii=False) + "\n")
        attempts = (previous.get("attempts", 0) if same_version else 0) + 1
        config = engine.config
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            if probe is None:
                from .real_model_probe import validate_loaded_backbone
                probe = validate_loaded_backbone
            result = probe(engine)
            if not result.get("passed"):
                raise AssertionError("KV-only preflight did not pass")
            data = dict(result, model=model, identity=identity, attempts=attempts,
                        probe_version=version, created=time.time())
        except Exception as exc:
            data = dict(
                passed=False, model=model, identity=identity, attempts=attempts, probe_version=version,
                created=time.time(), error=type(exc).__name__, message=str(exc),
                traceback=traceback.format_exc(),
            )
        finally:
            engine.config = config
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:
                torch.cuda.set_rng_state_all(gpu_rng)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(output)
        return "passed" if data["passed"] else "failed"
