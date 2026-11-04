#!/usr/bin/env python3
"""Run the original OLMo-style SD with read-only gain/transfer observers."""
from dataclasses import replace
from contextlib import nullcontext
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import time
import traceback

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_result(result, observations, entry):
    training = result["training"]
    selection = result["selection"]
    memory = result["standalone_cartridge"]["memory"]
    release = result["memory"]["full_cache_release_before_standalone"]
    if (training["objective"] != "on-policy KL(teacher || student)" or
        training["value_basis_source"] != "document" or training["selected_transfer_weight"] != 0 or
        training["student_rollout_limit"] != 8 or not training["teacher_conditioned_on_student_prefix"] or
        not training["student_rollouts_refreshed_each_step"] or not training["steps"] or
        len(observations) != len(training["steps"]) or
        selection["source_case_key"] != entry["key"] or
        selection["reference_answers_used"] is not False or selection["online_api_calls"] != 0 or
        result["self_study"]["reference_answers_used"] or result["self_study"]["student_answer_hint_used"] or
        result["teacher_answer_in_final_prompt"] or not result["backbone_unchanged"] or
        not result["original_document_KV_unchanged"] or
        release["full_cache_tensors_live_before_standalone"] != 0 or
        not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
        memory["additional_source_kv_bytes_before_decode"] != 0 or
        memory["additional_training_adapter_bytes_before_decode"] != 0 or
        result["generated_token_ids"] != result["standalone_cartridge"]["generated_token_ids"]):
        raise ValueError("Original SD or independent-cartridge diagnostic contract failed")
    for step, measured in zip(training["steps"], observations):
        if "unavailable" in measured:
            continue
        if not math.isclose(step["gain_gradient_norm"], measured["gradient"]["total_norm"],
                            rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError("The observed gain gradient differs from the training log")


def preflight(engine, *, value_alignment=False, zero_path_parity=False):
    from masked_ttt import MaskedTTT, Request
    from masked_ttt.kv_only import kv_only_cartridge
    from masked_ttt.request import prepare_layout
    from masked_ttt.selection import PreparedRequest
    from experiments.value_lr_physical.gain_transfer_probe import capture_gain_transfer
    from experiments.value_lr_physical.readout_transfer_probe import capture_first_answer_readout
    from experiments.value_lr_physical.value_alignment_probe import capture_value_alignment
    from experiments.value_lr_physical.zero_path_probe import capture_zero_path_parity

    document = "The old note is red. " + "Ordinary material. " * 30 + "The current note is blue."
    request = Request(document=document, question="What is the current note?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, item):
            layout = prepare_layout(backbone, item)
            return PreparedRequest(item, layout, [len(layout.chunks) // 2], dict(backend="synthetic"))

    original = engine.config
    try:
        engine.config = replace(original, chunk_tokens=64, steps=2, max_answer_steps=2,
            self_question_count=0, student_rollout_tokens=3, guard_enabled=False,
            selected_transfer_weight=0., value_basis_source="document")
        method = MaskedTTT(engine, Selector())
        with kv_only_cartridge():
            baseline = method.answer(request, time_budget_seconds=300)
        value_context = capture_value_alignment() if value_alignment else nullcontext({})
        parity_context = capture_zero_path_parity() if zero_path_parity else nullcontext({})
        with capture_gain_transfer() as observations, capture_first_answer_readout() as readout, \
             value_context as alignment, parity_context as parity, kv_only_cartridge():
            observed = method.answer(request, time_budget_seconds=300)
        if (baseline["teacher"]["tokens"] != observed["teacher"]["tokens"] or
            baseline["generated_token_ids"] != observed["generated_token_ids"] or
            len(baseline["training"]["steps"]) != len(observed["training"]["steps"]) or
            len(observations) != len(observed["training"]["steps"])):
            raise AssertionError("The observer changed the loaded-model trajectory")
        for a, b in zip(baseline["training"]["steps"], observed["training"]["steps"]):
            if a["student_rollout_tokens"] != b["student_rollout_tokens"] or a["target_positions"] != b["target_positions"]:
                raise AssertionError("The observer changed the student trajectory")
            for key in ("teacher_KL_before", "gain_gradient_norm", "value_gradient_norm"):
                if not math.isclose(a[key], b[key], rel_tol=1e-5, abs_tol=1e-6):
                    raise AssertionError("The observer changed the KL or gradients")
        if not any(s["gain_gradient_norm"] > 0 for s in observed["training"]["steps"]):
            raise AssertionError("The loaded model produced no gain gradient")
        memory = observed["standalone_cartridge"]["memory"]
        if (memory["additional_source_kv_bytes_before_decode"] != 0 or
            memory["additional_training_adapter_bytes_before_decode"] != 0 or
            observed["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0):
            raise AssertionError("The observer retained a training cache")
        statistics = readout.get("statistics", {"unavailable": "no_capture"})
        value_statistics = alignment.get("statistics", {"unavailable": "not_requested"})
        parity_statistics = parity.get("statistics", {"unavailable": "not_requested"})
        return dict(passed=True, gain_observer_neutral=True, full_and_source_released=True,
                    readout_available="unavailable" not in statistics,
                    readout_unavailable=statistics.get("unavailable"),
                    value_alignment_requested=value_alignment,
                    value_alignment_available=value_alignment and "unavailable" not in value_statistics,
                    value_alignment_unavailable=value_statistics.get("unavailable"),
                    zero_path_parity_requested=zero_path_parity,
                    zero_path_parity_available=zero_path_parity and "unavailable" not in parity_statistics,
                    zero_path_parity_unavailable=parity_statistics.get("unavailable"))
    finally:
        engine.config = original


def main():
    import argparse
    from time_budget import AllocationBudget, duration_seconds

    os.environ["MASKED_TTT_FIXED_CHAT_DATE"] = "2026-11-03"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--value-alignment", action="store_true")
    parser.add_argument("--zero-path-parity", action="store_true")
    args = parser.parse_args()
    root = args.job_root.resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The diagnostic requires one allocated GPU")
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY_FILE", None)
    manifest = json.loads((root / "manifest.json").read_text())
    if (len(manifest["case_ids"]) != 40 or len(set(manifest["case_ids"])) != 40 or
        manifest["models"] != ["qwen", "gpt"] or manifest["keep_fraction"] != .1 or
        manifest.get("value_alignment_diagnostic", False) is not args.value_alignment or
        manifest.get("zero_path_parity_diagnostic", False) is not args.zero_path_parity or
        manifest["reference_answers_used"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The fixed diagnostic protocol changed")
    import torch
    from masked_ttt import Backbone, FrozenJSONSelector, MaskedTTT, Request, TTTConfig
    from masked_ttt.decision_api import DecisionClient
    from masked_ttt.kv_only import kv_only_cartridge
    from experiments.value_lr_physical.gain_transfer_probe import capture_gain_transfer
    from experiments.value_lr_physical.readout_transfer_probe import capture_first_answer_readout
    from experiments.value_lr_physical.value_alignment_probe import capture_value_alignment
    from experiments.value_lr_physical.zero_path_probe import capture_zero_path_parity
    from experiments.value_lr_physical.report_gain_transfer import write_report

    def no_network(*_args, **_kwargs):
        raise AssertionError("Use cached Decision scores only")
    DecisionClient.__init__ = no_network
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one GPU is required")
    job = os.environ["SLURM_JOB_ID"]
    info = subprocess.run(["scontrol", "show", "job", job, "-o"], check=True, capture_output=True, text=True).stdout
    fields = dict(re.findall(r"(\S+?)=(\S+)", info))
    budget = AllocationBudget(root / "allocation_budget.json", 6 * 3600,
                              duration_seconds(fields["TimeLimit"]), duration_seconds(fields["RunTime"]))
    stopped = False
    terminated = False

    def handle(signum, *_):
        nonlocal stopped, terminated
        stopped = True
        terminated |= signum == signal.SIGTERM
        budget.record(active=True)

    signal.signal(signal.SIGUSR1, handle)
    signal.signal(signal.SIGTERM, handle)
    profile = json.loads((root / "olmo_profile.json").read_text())
    order = list(manifest["case_ids"])
    random.Random(manifest["random_order_seed"]).shuffle(order)
    identity = dict(job=job, gpu=torch.cuda.get_device_name(0),
                    node=os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
                    restart=os.environ.get("SLURM_RESTART_COUNT", "0"))
    for model in manifest["models"]:
        if all((root / "cases" / model / (key + ".json")).exists() for key in order):
            continue
        if stopped or time.monotonic() >= budget.deadline - 30:
            break
        spec = json.loads(Path(manifest["model_protocols"][model]).read_text())
        entries = {row["key"]: row for row in json.loads(Path(spec["index"]).read_text())}
        if any(key not in entries for key in order):
            raise ValueError("A diagnostic case is missing")
        config = replace(TTTConfig(**profile), max_context=131072, keep_fraction=.1,
                         practice_boundary_fallback=True, release_full_before_cartridge=True,
                         release_selected_source_before_decode=True,
                         selected_transfer_weight=0., value_basis_source="document")
        engine = Backbone.load(spec["model_path"], config, local_files_only=True)
        checks = preflight(engine, value_alignment=args.value_alignment,
                           zero_path_parity=args.zero_path_parity)
        save(root / "preflight" / (model + "_" + job + "_" + identity["restart"] + ".json"), checks)
        selector = FrozenJSONSelector(spec["shared_scores"], source_index=spec["index"], source_cases=spec["source_cases"])
        method = MaskedTTT(engine, selector)
        for key in order:
            path = root / "cases" / model / (key + ".json")
            if path.exists():
                continue
            if stopped or time.monotonic() >= budget.deadline - 30:
                break
            entry = entries[key]
            try:
                with open(spec["source_cases"], "rb") as source:
                    source.seek(entry["offset"])
                    case = json.loads(source.read(entry["bytes"]))
                if case["key"] != key or set(case) & {"answer", "answers", "references", "gold", "answer_spans"}:
                    raise ValueError("Evaluation answers must not enter the request")
                selector.bind_case(key)
                request = Request(document=case["document"], question=case["query"], prompt=case["prompt"],
                                  answer_prefix=case.get("answer_prefix", ""), max_new_tokens=case["max_new_tokens"])
                value_context = capture_value_alignment() if args.value_alignment else nullcontext({})
                parity_context = capture_zero_path_parity() if args.zero_path_parity else nullcontext({})
                with capture_gain_transfer() as observations, capture_first_answer_readout() as readout, \
                     value_context as alignment, parity_context as parity, kv_only_cartridge():
                    result = method.answer(request, time_budget_seconds=max(1, budget.deadline - time.monotonic()))
                validate_result(result, observations, entry)
                save(path, dict(key=key, model=model, task=entry["task"], length=entry["length_bucket"],
                    prediction=result["prediction"], generated_token_ids=result["generated_token_ids"],
                    selection=result["selection"], training=result["training"], observations=observations,
                    readout=readout.get("statistics", {"unavailable": "no_capture"}),
                    value_alignment=alignment.get("statistics", {"unavailable": "not_requested"}),
                    zero_path_parity=parity.get("statistics", {"unavailable": "not_requested"}),
                    memory=result["standalone_cartridge"]["memory"],
                    checks=dict(backbone_unchanged=result["backbone_unchanged"],
                        original_document_KV_unchanged=result["original_document_KV_unchanged"],
                        full_cache_released=result["memory"]["full_cache_release_before_standalone"][
                            "full_cache_tensors_live_before_standalone"] == 0,
                        teacher_answer_in_final_prompt=result["teacher_answer_in_final_prompt"]), **identity))
                completed_model = sum((root / "cases" / model / (item + ".json")).exists() for item in order)
                if completed_model % 5 == 0:
                    write_report(root)
                print(json.dumps(dict(model=model, case=key, complete=True)), flush=True)
                del request, result, observations, readout
            except TimeoutError:
                stopped = True
                break
            except Exception as exc:
                save(root / "errors" / model / (key + ".json"),
                     dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
                raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        del method, selector, engine
        gc.collect()
        torch.cuda.empty_cache()
    completed = {model: sum((root / "cases" / model / (key + ".json")).exists() for key in order)
                 for model in manifest["models"]}
    remaining = budget.record()
    save(root / "progress.json", dict(updated=time.time(), completed=completed,
        expected_per_model=40, remaining_compute_seconds=remaining, **identity))
    write_report(root)
    if all(value == 40 for value in completed.values()):
        save(root / "finished.json", dict(completed=completed, all_complete=True))
    elif not terminated and remaining > 900:
        subprocess.run(["scontrol", "requeue", job], check=True)


if __name__ == "__main__":
    main()
