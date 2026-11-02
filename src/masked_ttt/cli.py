"""Run the method on one user-provided document and question."""

import argparse
import json
from pathlib import Path

from . import (
    Backbone,
    DecisionConfig,
    FrozenJSONSelector,
    MaskedTTT,
    OpenAIChunkSelector,
    Request,
    TTTConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Masked TTT via Selective Self-Distillation")
    parser.add_argument(
        "--model", required=True, help="Qwen3, OLMo3, or GPT-OSS model path or model ID"
    )
    parser.add_argument(
        "--request", type=Path, required=True, help="JSON with document and question"
    )
    parser.add_argument(
        "--output", type=Path, help="Write prediction and method diagnostics as JSON"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--score-json-dir", type=Path, help="Precomputed source scores; disables all API calls"
    )
    parser.add_argument("--shard-gpus", type=int, default=1)
    parser.add_argument(
        "--api-key-file", help="Private credential file; alternatively use OPENAI_API_KEY"
    )
    parser.add_argument("--cache-dir", help="Local Decisions response cache and cost ledger")
    parser.add_argument("--max-api-cost", type=float, default=15.0)
    parser.add_argument("--keep-fraction", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument("--time-budget", type=float, default=None)
    args = parser.parse_args()
    # Unknown input fields (including gold answers) fail instead of silently
    # becoming adaptation inputs.
    request = Request(**json.loads(args.request.read_text()))
    config = TTTConfig(
        keep_fraction=args.keep_fraction, steps=args.steps, max_context=args.max_context
    )
    decision_options = dict(api_key_file=args.api_key_file, max_cost_usd=args.max_api_cost)
    if args.cache_dir:
        decision_options["cache_dir"] = args.cache_dir
    selector = (
        FrozenJSONSelector(args.score_json_dir)
        if args.score_json_dir
        else OpenAIChunkSelector(DecisionConfig(**decision_options))
    )
    backbone = Backbone.load(
        args.model,
        config,
        device=args.device,
        local_files_only=args.local_files_only,
        shard_gpus=args.shard_gpus,
    )
    result = MaskedTTT(backbone, selector).answer(request, time_budget_seconds=args.time_budget)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
