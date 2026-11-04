#!/usr/bin/env python3
"""Validate canonical evidence planning on fixed, unlabeled development cases.

This is a CPU-only feasibility check. It does not change training or inference,
does not load model weights, and does not include per-case text in its output.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer
from masked_ttt import Request, TTTConfig
from masked_ttt._layout import selected_read_plan
from masked_ttt.linked_selection import LinkedFrozenSelector
from experiments.value_lr_physical.audit_selected_tokenization import FinalChannelTokenizer, selected_chunk_positions
from experiments.value_lr_physical.canonical_evidence import build_canonical_evidence


def audit(workspace):
    root = workspace / "query_cartridges_linked_selection_20261104"
    manifest = json.loads((root / "manifest.json").read_text())
    cases = manifest["case_ids"]
    if (len(cases) != 200 or len(set(cases)) != 200 or manifest["models"] != ["qwen", "gpt"] or
        manifest["keep_fraction"] != .1 or manifest["reference_answers_used_in_inference"] is not False or
        manifest["online_api_calls"] != 0):
        raise ValueError("The frozen unlabeled development selection changed")
    profile = json.loads((workspace / "query_cartridges_value_lr_20261104/olmo_profile.json").read_text())
    models = {}
    for name in ("qwen", "gpt"):
        spec = json.loads(Path(manifest["model_protocols"][name]).read_text())
        tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], local_files_only=True)
        if name == "gpt":
            tokenizer = FinalChannelTokenizer(tokenizer, "2026-11-03")
        engine = SimpleNamespace(tokenizer=tokenizer, config=replace(TTTConfig(**profile), max_context=131072))
        index = {entry["key"]: entry for entry in json.loads(Path(spec["index"]).read_text())}
        selector = LinkedFrozenSelector(spec["shared_scores"])
        totals = Counter()
        with open(spec["source_cases"], "rb") as source:
            for key in cases:
                entry = index[key]
                source.seek(entry["offset"])
                case = json.loads(source.read(entry["bytes"]))
                if case["key"] != key or set(case) & {"answer", "answers", "references", "gold", "answer_spans"}:
                    raise ValueError("Unexpected answer-bearing source request")
                request = Request(document=case["document"], question=case["query"], prompt=case["prompt"],
                                  answer_prefix=case.get("answer_prefix", ""), max_new_tokens=case["max_new_tokens"])
                selector.bind_case(key)
                prepared = selector.prepare(engine, request)
                recorded = json.loads((root / "cases" / name / key / "result.json").read_text())["selection"]
                fields = ("source_case_key", "original_prompt_tokens", "working_prompt_tokens", "total_chunks",
                          "selected_original_chunks")
                if (any(prepared.metadata[field] != recorded[field] for field in fields) or
                    prepared.metadata["reference_answers_used"] or prepared.metadata["online_api_calls"] != 0):
                    raise ValueError("The local selected-evidence input differs from the completed model run")
                layout, selected = prepared.layout, prepared.selected
                separator = tokenizer.encode("\n\n", add_special_tokens=False)
                plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, selected, separator)
                mapping = selected_chunk_positions(layout, selected, plan, len(separator))
                try:
                    candidate = build_canonical_evidence(tokenizer, layout, selected, plan, mapping,
                                                         sink_tokens=engine.config.sink_tokens)
                except ValueError as exc:
                    totals["reject:" + str(exc)] += 1
                else:
                    totals["buildable"] += 1
                    totals["evidence_token_sequence_changed"] += int(candidate.ids != tuple(plan.ids))
                    totals["mixed_boundary_tokens"] += candidate.mixed_boundary_tokens
                    totals["source_document_tokens"] += plan.document_end - layout.initial
                    totals["canonical_document_tokens"] += candidate.document_end - layout.initial
                totals["cases"] += 1
        if totals["cases"] != 200:
            raise ValueError("Incomplete development cohort")
        models[name] = dict(totals)
    return dict(scope="200 source-only development cases per model, 50 sources at four lengths",
                model_weights_loaded=False, model_calls=0, online_api_calls=0,
                reference_answers_read=False, prediction_or_accuracy_measured=False, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
