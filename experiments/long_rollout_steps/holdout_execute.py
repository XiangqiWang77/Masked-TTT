#!/usr/bin/env python3
"""Run source-disjoint cases with the development-fixed original-SD step."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def model_paths(manifest):
    from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
    validate_partition(manifest["development_case_ids"], manifest["holdout_case_ids"])
    models = manifest["models"]
    steps = manifest["fixed_final_base_steps"]
    if (manifest["mode"] != "holdout" or manifest["case_count"] != 800 or
        set(manifest["case_ids"]) != set(manifest["holdout_case_ids"]) or
        not models or len(set(models)) != len(models) or any(m not in ("qwen", "gpt") for m in models) or
        set(models) != set(steps) or any(type(steps[m]) is not int or
        steps[m] not in (2, 4, 8) for m in models) or
        manifest["holdout_evaluated_at_selection"] is not False):
        raise ValueError("The holdout must use only development-fixed model steps")
    paths = {}
    for model in models:
        spec = json.loads(Path(manifest["model_protocols"][model]).read_text())
        path = str(Path(spec["model_path"]).resolve())
        if path in paths:
            raise ValueError("Model checkpoint paths must be unique")
        paths[path] = steps[model]
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    root = parser.parse_args().job_root.resolve()
    driver = root / "runtime/rollout_driver.py"
    if not driver.is_file() or not (root / "manifest.json").is_file():
        raise ValueError("The frozen runner and holdout manifest are required")
    manifest = json.loads((root / "manifest.json").read_text())
    paths = model_paths(manifest)
    spec = importlib.util.spec_from_file_location("frozen_holdout_runner", driver)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    implementation.CROSS = root.parent
    implementation.FROZEN = root / "runtime"
    implementation.REPO = REPO
    sys.path[:0] = [str(REPO / "src"), str(REPO), str(root / "runtime/helpers")]
    import masked_ttt as runtime
    import experiments.anchored_short_verifier.paired_controls as paired_controls
    from experiments.kv_only_cartridge.original_controls import kv_only_original_question_controls
    from experiments.long_rollout_steps.holdout_runner import fixed_step_runner
    from masked_ttt.linked_selection import LinkedFrozenSelector

    with fixed_step_runner(implementation, runtime, paired_controls, root=root,
            linked_selector=LinkedFrozenSelector, physical_controls=kv_only_original_question_controls,
            model_path_steps=paths):
        implementation.validate_manifest(manifest)
        implementation.main()


if __name__ == "__main__":
    main()
