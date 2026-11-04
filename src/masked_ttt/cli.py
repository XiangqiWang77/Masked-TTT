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


def load_method_config(path=None, **overrides):
    """Load the recorded profile and apply only explicit CLI overrides."""
    values = {}
    if path is not None:
        values = json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("The method config must be a JSON object")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return TTTConfig(**values)


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
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--score-json-dir", type=Path, help="Precomputed source scores; disables all API calls"
    )
    selection.add_argument("--online-decisions", action="store_true", help="Score source chunks with the Decision API")
    parser.add_argument("--case-id", help="Case ID in the precomputed source-score index")
    parser.add_argument("--source-index", type=Path,
                        help="Optional original-case index for exact document verification")
    parser.add_argument("--source-jsonl", type=Path,
                        help="Original indexed JSONL; must contain no evaluation answers")
    parser.add_argument("--shard-gpus", type=int, default=1)
    parser.add_argument(
        "--api-key-file", help="Private credential file; alternatively use OPENAI_API_KEY"
    )
    parser.add_argument("--cache-dir", help="Local Decisions response cache and cost ledger")
    parser.add_argument("--max-api-cost", type=float, default=15.0)
    parser.add_argument("--config-json", type=Path, help="Recorded TTTConfig JSON; explicit flags override it")
    parser.add_argument("--keep-fraction", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--gain-learning-rate", type=float)
    parser.add_argument("--value-learning-rate", type=float)
    parser.add_argument("--value-basis-source",choices=("document","selected"),default=None)
    parser.add_argument("--standalone-answer-format",choices=("original","teacher_requirements"),default=None)
    parser.add_argument("--align-student-teacher-prompt", action="store_true", default=None,
                        help="Use the Teacher's question-only formatting for the main SD path")
    parser.add_argument("--max-answer-steps", type=int)
    parser.add_argument("--teacher-max-new-tokens", type=int)
    parser.add_argument("--answer-kl-stop", type=float)
    parser.add_argument("--selected-transfer-weight", type=float,
                        help="Weight for full-to-selected on-policy KL; 0 preserves the archived method")
    parser.add_argument("--release-full-before-cartridge",action="store_true",default=None,
                        help="Release full-document KV before standalone cartridge decoding")
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--context-reserve-tokens", type=int,
                        help="Minimum room reserved for the generated answer")
    parser.add_argument("--time-budget", type=float, default=None)
    args = parser.parse_args()
    # Unknown input fields (including gold answers) fail instead of silently
    # becoming adaptation inputs.
    request = Request(**json.loads(args.request.read_text()))
    try:
        config = load_method_config(args.config_json, keep_fraction=args.keep_fraction, steps=args.steps,
            gain_learning_rate=args.gain_learning_rate,value_learning_rate=args.value_learning_rate,
            value_basis_source=args.value_basis_source,
            standalone_answer_format=args.standalone_answer_format,
            align_student_teacher_prompt=args.align_student_teacher_prompt,
            max_context=args.max_context, context_reserve_tokens=args.context_reserve_tokens,
            max_answer_steps=args.max_answer_steps,teacher_max_new_tokens=args.teacher_max_new_tokens,
            answer_KL_stop=args.answer_kl_stop,selected_transfer_weight=args.selected_transfer_weight,
            release_full_before_cartridge=args.release_full_before_cartridge)
    except (OSError,ValueError,TypeError) as exc:
        parser.error('Invalid method configuration: '+str(exc))
    decision_options = dict(api_key_file=args.api_key_file, max_cost_usd=args.max_api_cost)
    if args.cache_dir:
        decision_options["cache_dir"] = args.cache_dir
    if (args.source_index is None) != (args.source_jsonl is None):
        parser.error('--source-index and --source-jsonl must be supplied together')
    if args.online_decisions and args.source_index is not None:
        parser.error('source verification applies to precomputed scores')
    if args.score_json_dir:
        if not args.case_id:
            parser.error('--case-id is required with --score-json-dir')
        selector = (FrozenJSONSelector(args.score_json_dir, source_index=args.source_index,
                                       source_cases=args.source_jsonl)
                    if args.source_index is not None else FrozenJSONSelector(args.score_json_dir))
        selector.bind_case(args.case_id)
    else:
        selector = OpenAIChunkSelector(DecisionConfig(**decision_options))
    backbone = Backbone.load(
        args.model,
        config,
        device=args.device,
        local_files_only=args.local_files_only,
        shard_gpus=args.shard_gpus,
    )
    # Keep the compact choice in this request rather than patching the shared
    # pipeline. Explicit source-retention settings preserve the debug path.
    compact = (config.standalone_cartridge_enabled and config.release_full_before_cartridge and
               config.release_selected_source_before_decode)
    result = MaskedTTT(backbone, selector).answer(
        request, time_budget_seconds=args.time_budget, compact_standalone=compact)
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
