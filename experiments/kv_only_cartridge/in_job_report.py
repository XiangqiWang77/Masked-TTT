#!/usr/bin/env python3
"""Summarize real-model preflight artifacts; never queries the scheduler."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

REQUIRED = ("same_teacher_tokens", "same_student_rollouts", "same_training_gradients",
            "same_standalone_tokens", "same_standalone_logits", "same_request_chat_date",
            "no_full_kv_or_extra_adapter_at_decode", "original_benchmark_runtime_unchanged")


def collect(directory):
    directory = Path(directory)
    if not (directory / "ready.json").is_file():
        raise ValueError("The independent preflight was not enabled")
    models = {}
    for model, expected_type in (("qwen", "qwen3"), ("gpt", "gpt_oss")):
        path = directory / (model + ".json")
        if not path.exists():
            models[model] = dict(state="waiting")
            continue
        data = json.loads(path.read_text())
        if data.get("passed") is not True:
            models[model] = dict(state="failed", error_type=data.get("error"),
                                 attempts=data.get("attempts"))
            continue
        gpu = data.get("identity", {}).get("gpu")
        valid = (data.get("model") == model and data.get("model_type") == expected_type and
                 isinstance(gpu, str) and bool(gpu) and "cpu" not in gpu.lower() and
                 all(data.get(key) is True for key in REQUIRED) and
                 data.get("online_api_calls") == 0 and data.get("reference_answers_used") is False)
        models[model] = (dict(state="passed", gpu=gpu, attempts=data.get("attempts")) if valid else
                         dict(state="invalid", reason="missing real-GPU or method evidence"))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), models=models,
                online_api_calls=0, new_gpu_jobs=0, scheduler_queries=0,
                scope="synthetic deployment parity; not a benchmark accuracy result")


def render(report):
    lines = ["# Request-local independent cartridge: real-model preflight", "",
             "| Model | State | GPU |", "|---|---|---|"]
    for model, item in report["models"].items():
        lines.append(f"| {model} | {item['state']} | {item.get('gpu', '—')} |")
    lines += ["", "A pass requires matching Teacher tokens, student rollouts, joint gradients, "
              "standalone tokens and logits, together with release of full KV and extra adapter state. "
              "This synthetic preflight does not measure answer accuracy.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    with (args.directory / "report.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = collect(args.directory)
        for name, content in (("latest.json", json.dumps(result, indent=2) + "\n"),
                              ("latest.md", render(result))):
            path = args.directory / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content)
            temporary.replace(path)
    print(json.dumps(result["models"]))


if __name__ == "__main__":
    main()
