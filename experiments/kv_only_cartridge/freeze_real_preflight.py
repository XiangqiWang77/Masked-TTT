#!/usr/bin/env python3
"""Freeze verified real-GPU request-local deployment checks without tokens."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.kv_only_cartridge.in_job_report import REQUIRED, collect


def freeze(qwen, gpt, gpt_status):
    core = tuple(key for key in REQUIRED if key not in ("same_checkpoint_rope", "native_checkpoint_config_checked"))
    result = {}
    for model, data, version, kind, requirements in (
        ("qwen", qwen, 3, "qwen3", core), ("gpt", gpt, 4, "gpt_oss", REQUIRED),
    ):
        gpu = data.get("identity", {}).get("gpu")
        if (data.get("model") != model or data.get("model_type") != kind or
            data.get("probe_version") != version or data.get("passed") is not True or
            not isinstance(gpu, str) or "cpu" in gpu.lower() or not gpu or
            not all(data.get(key) is True for key in requirements) or
            data.get("generic_teacher_answer_in_prompt") is not False or
            data.get("online_api_calls") != 0 or data.get("reference_answers_used") is not False):
            raise ValueError("A real-model preflight is incomplete or invalid: " + model)
        result[model] = dict(state="passed", model_type=kind, gpu=gpu, probe_version=version,
            created_utc=datetime.fromtimestamp(data["created"], timezone.utc).isoformat(),
            same_teacher_tokens=True, same_student_rollouts=True, same_training_gradients=True,
            same_standalone_tokens_and_logits=True, source_and_training_state_released=True,
            generic_tokens_match_after_release=True, teacher_answer_in_final_prompt=False,
            native_rope_on_gpu_checked=version >= 4)
    if gpt_status.get("state") != "passed" or gpt_status.get("gpu") != result["gpt"]["gpu"]:
        raise ValueError("The shared GPT v4 preflight report did not validate")
    return dict(models=result, synthetic_request=True, quality_not_measured=True,
                reference_answers_used=False, online_api_calls=0, new_gpu_jobs=0,
                qwen_v4_gpu_rope_check_pending=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v3 = args.workspace / "query_cartridges_request_local_real_probe_v3_20261104"
    v4 = args.workspace / "query_cartridges_request_local_real_probe_v4_20261104"
    qwen, gpt = (json.loads(path.read_text()) for path in (v3 / "qwen.json", v4 / "gpt.json"))
    result = freeze(qwen, gpt, collect(v4)["models"]["gpt"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["models"], indent=2))


if __name__ == "__main__":
    main()
