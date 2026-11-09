"""Prepare the closed 160-case Contextual Cartridges development gate.

The case partition and all optimizer/model settings come from v24.  Paired
inputs are rebuilt from the fresh entailment-preserving query-specific data,
and Selective SD uses the pre-existing original task-conditioned support gate.
No v24 target, assessment, Cartridge, arm prediction, or benchmark answer is
reused.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import zlib

from experiments.run_syn_sd_factorial import input_rows, read, write


VERSION = "official-contextual-cartridges-tracin-selective-v26"


def prepare(previous_config, dataset_root, output, hours):
    old = read(previous_config)
    dataset_root, output = Path(dataset_root), Path(output)
    if output.exists():
        raise FileExistsError(f"Do not overwrite an experiment: {output}")
    readiness = read(dataset_root / "synthetic_dataset_export_progress.json")
    if not readiness.get("complete") or readiness["counts"]["completed_data_cases"] != 1000:
        raise ValueError("The fresh 1000-case query-specific dataset is not complete")
    now = time.time()
    config = deepcopy(old)
    config.update(
        protocol=VERSION,
        output=str(output),
        dataset_root=str(dataset_root),
        paired_input_dir=str(output / "paired_inputs"),
        selective_gate="original_task_conditioned_support_required",
        tracin_selective_sd=True,
        tracin_threshold=0.0,
        tracin_checkpoints=["initial_cartridge", "vanilla_SD"],
        selective_checkpoint_envelope=False,
        selective_validation_policy="no_heldout_tracin_training_dynamics",
        phase_control_path=str(output / "phase.json"),
        started_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        deadline_epoch=now + hours * 3600,
        deadline_utc=datetime.fromtimestamp(now + hours * 3600, timezone.utc).isoformat(),
        previous_diagnostic_run=old["output"],
        old_teacher_targets_reused=False,
        old_selective_assessments_reused=False,
        old_cartridges_or_predictions_reused=False,
        benchmark_answers_used_for_training_or_selection=False,
        production_phase="closed_development_gate",
        workers=4,
        paired_cases=160,
        deadline_from_slurm_allocation=True,
    )
    development = [item for item in config["cases"]
                   if item.get("analysis_split") == "development"]
    if len(development) != int(config["development_cases"]):
        raise ValueError("Expected the frozen 160-case development split")
    config["cases"] = development
    config.pop("array_job_id", None)
    output.mkdir(parents=True)
    vanilla_root = Path(config.get(
        "vanilla_dataset_root", config["vanilla_generation_root"]))
    paired_counts = {"train": 0, "validation": 0}
    for item in config["cases"]:
        key = item["key"]
        specific = read(dataset_root / "dataset_cases" / f"{key}.json")
        vanilla = read(vanilla_root / "dataset_cases" / f"{key}.json")
        if specific.get("complete") is not True or vanilla.get("complete") is not True:
            raise ValueError(f"Incomplete paired dataset: {key}")
        counts = {
            split: min(int(specific["counts"][split]), int(vanilla["counts"][split]))
            for split in ("train", "validation")
        }
        if any(value < 1 for value in counts.values()):
            raise ValueError(f"A paired split is empty: {key}")
        item["specific_marker"] = str(dataset_root / "dataset_cases" / f"{key}.json")
        item["paired_split_counts"] = counts
        item["original"] = {
            "files": [vanilla["paths"]["original_train"], vanilla["paths"]["original_validation"]],
            "source": vanilla["paths"]["source"],
            "root": str(vanilla_root),
            "recipe": "official_query_agnostic_regeneration_common_prefix_matched_to_specific",
            "generation_provenance": "fresh_full1000",
        }
        seed = (config["seed"] + zlib.crc32(key.encode())) % (2**31)
        rows, sampling = input_rows(item, config, seed)
        for split in ("train", "validation"):
            if len(rows["vanilla_syns"][split]) != len(rows["specific_syns"][split]):
                raise ValueError(f"Unmatched question budget for {key}/{split}")
            paired_counts[split] += len(rows["specific_syns"][split])
        write(
            output / "paired_inputs" / f"{key}.json",
            {
                "case": key,
                "datasets": rows,
                "vanilla_sampled_indices": sampling,
                "benchmark_answers_used": False,
                "synthetic_answer_labels_used": False,
            },
        )
    write(
        output / "phase.json",
        {
            "phase": "development",
            "holdout_enabled": False,
            "reason": "the 840-case holdout stays closed until every frozen development gate passes",
        },
    )
    write(
        output / "success_criteria.json",
        {
            "development_cases": config["development_cases"],
            "query_specific_dataset_positive": (
                "specific_syns_vanilla_sd > vanilla_syns_vanilla_sd"
            ),
            "selective_sd_on_each_fixed_dataset_positive": True,
            "strict_oracle_escape": (
                "within full-Teacher-wrong and initial-Cartridge-wrong cases, "
                "Selective SD has at least one fix and fixes minus regressions > 0"
            ),
            "selective_checkpoint_rule": (
                "retain a source-admitted Teacher trajectory iff its summed signed "
                "TracIn influence over initial and Vanilla-SD checkpoints is > 0"
            ),
            "combined_contextual_cartridges_positive": True,
            "holdout_stays_closed_until_development_passes": True,
            "benchmark_answers_used_for_training_or_selection": False,
        },
    )
    write(
        output / "protocol_audit.json",
        {
            "schema": VERSION,
            "previous_diagnostic_run": old["output"],
            "corrected_implementation_failures": [
                "answer extraction previously treated possibility as certainty",
                "the v24 gate previously treated non-contradiction plus a quote as support",
            ],
            "corrections": [
                "fresh answer-first generation requires logical entailment and preserves modality",
                "two reverse questions per answer reduce repeated amplification of one answer anchor",
                "Selective SD uses the existing original task-conditioned support verification chain",
                "apply TracIn on native K,V gradients and retain only positive-influence "
                "source-admitted trajectories",
            ],
            "unchanged": [
                "frozen Qwen3-4B backbone",
                "native trainable Cartridge K,V only",
                "original sparse Teacher KL self-distillation",
                "ordinary-SD warmup before selective consolidation",
                "exact ordinary-SD fallback when no selective signal is accepted",
                "matched optimizer updates and identical initial Cartridge",
                "official BABILong generation and scoring",
            ],
            "no_attention_change": True,
            "no_g_B_LoRA": True,
            "no_answer_NLL": True,
            "no_RL_bilevel_or_counterfactual_SD": True,
            "benchmark_answers_used": False,
            "tracin_reference_is_full_ordinary_SD_training_objective": True,
            "tracin_heldout_set_used": False,
            "tracin_threshold": 0.0,
            "tracin_is_first_order_influence_not_correctness_proof": True,
            "v24_targets_assessments_checkpoints_predictions_reused": False,
        },
    )
    write(
        output / "input_inventory.json",
        {
            "paired_cases": len(config["cases"]),
            "development_cases": config["development_cases"],
            "holdout_cases": config["holdout_cases"],
            "paired_question_counts_per_dataset": paired_counts,
            "query_specific_root": str(dataset_root),
            "vanilla_root": str(vanilla_root),
            "benchmark_answers_used": False,
        },
    )
    write(output / "config.json", config)
    return output / "config.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=12.0)
    args = parser.parse_args()
    path = prepare(args.previous_config, args.dataset_root, args.output, args.hours)
    print(json.dumps({"config": str(path), "protocol": VERSION}))


if __name__ == "__main__":
    main()
