"""Run a synthetic physical-transfer check inside an existing allocation.

The benchmark's model is borrowed only briefly. No API or benchmark answers
are used, and its configuration, RNG state and temporary hooks are restored.
"""
import gc
import fcntl
import importlib.util
import json
from pathlib import Path
import time

import torch


def _load_probe(root):
    path = Path(root) / "probe.py"
    spec = importlib.util.spec_from_file_location("query_cartridges_physical_counterfactual", path)
    if spec is None or spec.loader is None:
        raise ValueError("The frozen counterfactual probe is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_if_needed(engine, model, identity, *, _root=None):
    """Run at most once per model, with no impact on benchmark training."""
    if model not in ("qwen", "gpt") or engine.device.type != "cuda":
        raise ValueError("A supported backbone on an allocated GPU is required")
    root = Path(__file__).resolve().parent if _root is None else Path(_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / (model + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / (model + "_result.json")
        if path.is_file() and json.loads(path.read_text()).get("probe_validated"):
            return "previously_validated"
        failure = root / (model + "_failure.json")
        attempts = json.loads(failure.read_text()).get("attempts", 0) if failure.is_file() else 0
        if attempts >= 2:
            return "two_previous_failures"
        config, cpu_rng = engine.config, torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state_all()
        try:
            data = _load_probe(root).run_pair(engine, steps=2, max_new_tokens=8, rollout_tokens=8,
                                              time_budget_seconds=90)
            if (not data.get("selected_inputs_identical") or not data.get("candidates_used_after_training_only") or
                data.get("teacher_answer_in_final_prompt") or not data.get("whole_cache_released_before_cartridge") or
                not data.get("source_cache_released_before_scoring") or
                not data.get("training_adapter_released_before_scoring") or
                not data.get("probabilities_from_materialized_kv")):
                raise AssertionError("The physical counterfactual contract changed")
            data.update(probe_validated=True, model=model, gpu=identity.get("gpu"),
                timestamp=time.time(), reference_answers_used=False, online_api_calls=0,
                extra_gpu_jobs=0, training_runtime_changed=False)
            _atomic(path, data)
            return "validated"
        except Exception as exc:
            _atomic(failure, dict(attempts=attempts+1, error_type=type(exc).__name__, timestamp=time.time()))
            raise
        finally:
            engine.config = config
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(gpu_rng)
            gc.collect()
            torch.cuda.empty_cache()
