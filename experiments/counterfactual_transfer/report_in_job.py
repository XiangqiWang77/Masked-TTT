#!/usr/bin/env python3
"""Summarize optional synthetic physical-counterfactual results without a model."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def collect(root):
    root=Path(root);models={}
    for model in ("qwen","gpt"):
        result=root/(model+"_result.json");failure=root/(model+"_failure.json")
        if not result.is_file():
            failed=json.loads(failure.read_text()) if failure.is_file() else {}
            models[model]=dict(status="failed" if failed.get("attempts",0)>=2 else "pending",
                               attempts=failed.get("attempts",0),error_type=failed.get("error_type"))
            continue
        data=json.loads(result.read_text())
        required=("selected_inputs_identical","candidates_used_after_training_only",
            "whole_cache_released_before_cartridge","source_cache_released_before_scoring",
            "training_adapter_released_before_scoring","probabilities_from_materialized_kv")
        delta=data.get("margin_difference_garden_minus_office_context")
        error=data.get("dynamic_physical_max_abs_logprob_difference")
        zero=data.get("zero_cache_max_logprob_difference")
        norms=data.get("state_difference_norms")
        if (not data.get("probe_validated") or data.get("model") != model or
            not all(data.get(key) is True for key in required) or
            data.get("reference_answers_used") is not False or data.get("online_api_calls") != 0 or
            data.get("extra_gpu_jobs") != 0 or data.get("training_runtime_changed") is not False or
            type(delta) not in (int,float) or not math.isfinite(delta) or
            type(error) not in (int,float) or not math.isfinite(error) or error < 0 or
            type(zero) not in (int,float) or not math.isfinite(zero) or not 0<=zero<=1e-4 or
            not isinstance(norms,dict) or any(type(norms.get(key)) not in (int,float) or
                not math.isfinite(norms[key]) or norms[key]<0 for key in ("gain","value_coefficients","fixed_value_basis")) or
            any(type(data.get(key)) is not bool for key in ("full_baseline_mentions_expected_for_both",
                                                            "cartridge_mentions_expected_for_both"))):
            models[model]=dict(status="invalid")
            continue
        models[model]=dict(status="valid",gpu=data.get("gpu"),timestamp=data.get("timestamp"),
            zero_cache_max_logprob_difference=data.get("zero_cache_max_logprob_difference"),
            dynamic_physical_max_abs_logprob_difference=error,
            margin_difference_garden_minus_office_context=delta,
            full_baseline_mentions_expected_for_both=data.get("full_baseline_mentions_expected_for_both"),
            cartridge_mentions_expected_for_both=data.get("cartridge_mentions_expected_for_both"),
            state_difference_norms=data.get("state_difference_norms"))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),models=models,
        all_models_valid=all(row["status"]=="valid" for row in models.values()),
        synthetic_intervention=True,reference_answers_used=False,online_api_calls=0,extra_gpu_jobs=0)


def render(data):
    lines=["# Physical Query Cartridge counterfactual probe", "",
        "| Model | Status | Full answers both | Cartridge answers both | Garden − office margin difference |",
        "|---|---|---|---|---:|"]
    for model,row in data["models"].items():
        if row["status"]=="valid":
            lines.append(f"| {model} | valid | {row['full_baseline_mentions_expected_for_both']} | "
                f"{row['cartridge_mentions_expected_for_both']} | "
                f"{row['margin_difference_garden_minus_office_context']:.6f} |")
        else:
            lines.append(f"| {model} | {row['status']} | — | — | — |")
    lines += ["", "The selected evidence and question stay identical; only an unselected location changes. "
        "The margin is measured from the physically materialized cartridge after full KV, selected source KV "
        "and training state have been released. A positive difference is directional sensitivity, not a "
        "general accuracy result. This is one synthetic pair, with no benchmark labels or online API.", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root",required=True,type=Path)
    args=parser.parse_args();root=args.results_root.resolve()
    data=collect(root)
    for name,text in (("latest.json",json.dumps(data,indent=2,allow_nan=False)+"\n"),
                      ("latest.md",render(data))):
        target=root/name;tmp=target.with_suffix(target.suffix+".tmp")
        tmp.write_text(text);tmp.replace(target)
    print(json.dumps(dict(all_models_valid=data["all_models_valid"],
                          models={k:v["status"] for k,v in data["models"].items()})))


if __name__=="__main__":main()
