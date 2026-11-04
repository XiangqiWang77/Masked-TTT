#!/usr/bin/env python3
"""Compare old paired and new physical runtimes on synthetic frozen models.

The parent launches isolated workers so the two masked_ttt packages never
coexist in one interpreter. No corpus, labels, network or GPU is required.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]


def worker(runtime, architecture, value_lr, old_root, base_steps):
    from dataclasses import fields, replace
    from masked_ttt import MaskedTTT, Request
    from masked_ttt.request import prepare_layout
    from masked_ttt.selection import PreparedRequest
    from test_materialization_equivalence import gpt_gqa_engine
    from test_standalone import make_engine

    engine = make_engine() if architecture == "qwen" else gpt_gqa_engine()
    engine.eos_ids = {-1}
    overrides = dict(self_question_count=0, guard_enabled=False, steps=base_steps,
        student_rollout_tokens=3, value_learning_rate=value_lr,
        selected_transfer_weight=0., value_basis_source="document", release_full_before_cartridge=True)
    if any(item.name == "release_selected_source_before_decode" for item in fields(engine.config)):
        overrides["release_selected_source_before_decode"] = True
    engine.config = replace(engine.config, **overrides)
    request = Request(document="The old note is red. The current note is blue. " * 6,
                      question="What color is the current note?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [1, len(layout.chunks)-2], dict(backend="synthetic"))

    if runtime == "old":
        sys.path.insert(0, str(old_root))
        from paired_prompt import paired_prompt_control
        context = paired_prompt_control()
    elif runtime == "new":
        from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
        context = kv_only_original_question_controls()
    else:
        raise ValueError("Unsupported runtime")
    with context:
        result = MaskedTTT(engine, Selector()).answer(request)
    standalone = result["standalone_cartridge"]
    if runtime == "old":
        tokens = standalone["paired_prompt_control"]["trained_original"]["tokens"]
    else:
        tokens = result["generated_token_ids"]
        memory = standalone["memory"]
        if (not memory["kv_only_materialized"] or memory["additional_source_kv_bytes_before_decode"] != 0 or
            memory["additional_training_adapter_bytes_before_decode"] != 0):
            raise AssertionError("The new standalone cache was not physically released")
    fields_to_compare = ("teacher_KL_before", "gain_gradient_norm", "value_gradient_norm", "gradient_norm",
                         "student_rollout_tokens", "target_positions", "question_path")
    return dict(runtime=runtime, architecture=architecture, value_learning_rate=value_lr, base_steps=base_steps,
        teacher_tokens=result["teacher"]["tokens"], final_tokens=tokens,
        full_prediction=result["self_judge"]["adapted_prediction"],
        full_cache_released=result["memory"]["full_cache_release_before_standalone"]
            ["full_cache_tensors_live_before_standalone"] == 0,
        weights_unchanged=result["backbone_unchanged"], cache_unchanged=result["original_document_KV_unchanged"],
        objective=result["training"]["objective"],
        steps=[{key: row[key] for key in fields_to_compare} for row in result["training"]["steps"]])


def compare(old, new, *, atol=2e-6, rtol=2e-5):
    for key in ("architecture", "value_learning_rate", "base_steps", "teacher_tokens", "final_tokens", "full_prediction",
                "full_cache_released", "weights_unchanged", "cache_unchanged", "objective"):
        if old[key] != new[key]:
            raise ValueError("Archived and physical runtimes differ on " + key)
    if not all(new[key] for key in ("full_cache_released", "weights_unchanged", "cache_unchanged")):
        raise ValueError("The frozen-request contract did not hold")
    if len(old["steps"]) != len(new["steps"]) or not old["steps"]:
        raise ValueError("The runtimes used a different update budget")
    max_abs = 0.
    for left, right in zip(old["steps"], new["steps"]):
        for key in ("student_rollout_tokens", "target_positions", "question_path"):
            if left[key] != right[key]:
                raise ValueError("The on-policy training trajectory differs on " + key)
        for key in ("teacher_KL_before", "gain_gradient_norm", "value_gradient_norm", "gradient_norm"):
            a, b = left[key], right[key]
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
                raise ValueError("The training signal differs on " + key)
            max_abs = max(max_abs, abs(a-b))
    return dict(architecture=new["architecture"], value_learning_rate=new["value_learning_rate"], base_steps=new["base_steps"],
        same_teacher_tokens=True, same_student_prefixes=True, same_final_tokens=True,
        max_training_signal_absolute_difference=max_abs, independent_cartridge=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--worker", choices=("old", "new"))
    parser.add_argument("--architecture", choices=("qwen", "gpt"))
    parser.add_argument("--value-lr", type=float)
    parser.add_argument("--base-steps", type=int, choices=(2, 8), default=2)
    args = parser.parse_args()
    if args.worker:
        if args.architecture is None or args.value_lr not in (.0001, .001):
            raise ValueError("Worker architecture and predeclared LR are required")
        print(json.dumps(worker(args.worker, args.architecture, args.value_lr, args.old_root, args.base_steps)))
        return
    results = []
    for architecture in ("qwen", "gpt"):
        for rate in (.0001, .001):
            pair = []
            for runtime in ("old", "new"):
                src = args.old_root / "runtime/src" if runtime == "old" else REPO / "src"
                env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(src), str(REPO / "tests"), str(REPO))),
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MASKED_TTT_FIXED_CHAT_DATE="2026-11-03")
                command = [str(args.python), str(Path(__file__).resolve()), "--old-root", str(args.old_root),
                    "--python", str(args.python), "--worker", runtime, "--architecture", architecture,
                    "--value-lr", str(rate), "--base-steps", str(args.base_steps)]
                result = subprocess.run(command, env=env, check=False, capture_output=True, text=True, timeout=120)
                if result.returncode:
                    raise RuntimeError(f"{runtime} {architecture} worker failed:\n" + result.stderr[-4000:])
                pair.append(json.loads(result.stdout.strip().splitlines()[-1]))
            results.append(compare(*pair))
    print(json.dumps(dict(synthetic=True, online_api_calls=0, reference_answers_used=False, results=results), indent=2))


if __name__ == "__main__":
    main()
