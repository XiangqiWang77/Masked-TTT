#!/usr/bin/env python3
"""Audit source-character coverage of selected model-native chunk tokens.

Uses cached Decision selection and local tokenizers on answer-free development
requests. No model weights, inference, API, reference answers or hashes are
used. The output contains aggregates only.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer
from masked_ttt import Request, TTTConfig
from masked_ttt.linked_selection import LinkedFrozenSelector
from experiments.value_lr_physical.audit_selected_tokenization import FinalChannelTokenizer


def character_coverage(document, spans, chunks, selected, offsets, document_start):
    if (not document or len(spans) != len(chunks) or not selected or
        list(selected) != sorted(set(selected)) or any(not 0 <= index < len(chunks) for index in selected)):
        raise ValueError("Expected ordered selected source spans and model-token chunks")
    requested, covered = bytearray(len(document)), bytearray(len(document))
    for index in selected:
        left, right = spans[index]
        if not 0 <= left < right <= len(document):
            raise ValueError("A selected source span is outside the document")
        requested[left:right] = b"\1" * (right-left)
    selected_tokens = [token for index in selected for token in chunks[index]]
    if selected_tokens != sorted(set(selected_tokens)):
        raise ValueError("Selected model tokens must be disjoint and ordered")
    for token in selected_tokens:
        if not 0 <= token < len(offsets):
            raise ValueError("A selected token has no character offset")
        left, right = offsets[token]
        if not 0 <= left <= right:
            raise ValueError("A selected token has an invalid character offset")
        left, right = max(0, left-document_start), min(len(document), right-document_start)
        if left < right:
            covered[left:right] = b"\1" * (right-left)
    requested_nonspace = missing_nonspace = extra_nonspace = 0
    for i, char in enumerate(document):
        if char.isspace():
            continue
        if requested[i]:
            requested_nonspace += 1
            missing_nonspace += not covered[i]
        elif covered[i]:
            extra_nonspace += 1
    missing_spans = 0
    for index in selected:
        left, right = spans[index]
        missing_spans += any(not document[i].isspace() and not covered[i] for i in range(left, right))
    return dict(selected_chunks=len(selected), selected_tokens=len(selected_tokens),
                requested_nonspace=requested_nonspace, missing_nonspace=missing_nonspace,
                extra_nonspace=extra_nonspace, missing_selected_spans=missing_spans)


def summarize(rows):
    return dict(cases=len(rows), cases_missing_selected_nonspace=sum(row["missing_nonspace"] > 0 for row in rows),
        cases_including_adjacent_nonspace=sum(row["extra_nonspace"] > 0 for row in rows),
        total_selected_chunks=sum(row["selected_chunks"] for row in rows),
        total_missing_selected_spans=sum(row["missing_selected_spans"] for row in rows),
        total_requested_nonspace=sum(row["requested_nonspace"] for row in rows),
        total_missing_nonspace=sum(row["missing_nonspace"] for row in rows),
        total_adjacent_nonspace=sum(row["extra_nonspace"] for row in rows),
        maximum_missing_nonspace_per_case=max((row["missing_nonspace"] for row in rows), default=0),
        missing_nonspace_histogram=dict(sorted(Counter(str(row["missing_nonspace"]) for row in rows).items(),
                                                key=lambda item: int(item[0]))))


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
        rows = []
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
                    prepared.metadata["reference_answers_used"] or prepared.metadata["online_api_calls"] != 0 or
                    not prepared.metadata["complete_original_document"]):
                    raise ValueError("The source selection differs from the completed model run")
                text = tokenizer.apply_chat_template([dict(role="user", content=request.prompt)],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False) + request.answer_prefix
                start = text.find(request.document)
                if start < 0 or text.find(request.document, start+1) >= 0:
                    raise ValueError("The original document must occur exactly once in the prompt")
                encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                if encoded["input_ids"] != prepared.layout.ids:
                    raise ValueError("The native tokenizer and selected layout differ")
                rows.append(character_coverage(request.document, prepared.layout.spans, prepared.layout.chunks,
                    prepared.selected, encoded["offset_mapping"], start))
        models[name] = summarize(rows)
    return dict(scope="200 source-only development cases per model, 50 sources at four lengths",
        model_weights_loaded=False, model_calls=0, online_api_calls=0, reference_answers_used=False,
        predictions_or_accuracy_measured=False, covers_selected_source_characters_not_support_annotations=True,
        models=models)


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
