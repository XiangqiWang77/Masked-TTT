#!/usr/bin/env python3
"""Audit selected-evidence token splicing using local tokenizers and no model."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
from types import SimpleNamespace

from transformers import AutoTokenizer

from masked_ttt import Request, TTTConfig
from masked_ttt._layout import selected_read_plan
from masked_ttt.linked_selection import LinkedFrozenSelector


class FinalChannelTokenizer:
    """Mirror the production GPT-OSS final-channel template without loading torch."""
    def __init__(self, tokenizer, fixed_date):
        self.tokenizer, self.fixed_date = tokenizer, fixed_date

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        tokenize = kwargs.pop("tokenize", False)
        kwargs.pop("enable_thinking", None)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, reasoning_effort="low", **kwargs)
        prefix = "<|start|>system<|message|>"
        if text.startswith(prefix):
            end = text.find("<|end|>", len(prefix))
            if end < 0:
                raise ValueError("Unexpected GPT-OSS system-message boundary")
            text = re.sub(r"(?m)(?<=Current date: )\d{4}-\d{2}-\d{2}(?=\n)",
                          self.fixed_date, text[:end], count=1) + text[end:]
        if kwargs.get("add_generation_prompt"):
            if not text.endswith("<|start|>assistant"):
                raise ValueError("Unexpected GPT-OSS generation boundary")
            text += "<|channel|>final<|message|>"
        return self.tokenizer.encode(text, add_special_tokens=False) if tokenize else text


def selected_chunk_positions(layout, selected, plan, separator_length):
    """Reconstruct the production mapping with only its source-position metadata."""
    if separator_length < 1:
        raise ValueError("A nonempty separator is required")
    mapping, cursor, previous = {}, layout.initial, layout.initial - 1
    for original in plan.original_document_positions:
        if original != previous + 1:
            cursor += separator_length
        mapping[original] = cursor
        cursor += 1
        previous = original
    if cursor != plan.document_end:
        raise ValueError("Selected cache position mapping is inconsistent")
    return [[mapping[position] for position in layout.chunks[index]] for index in selected]


def inspect_plan(tokenizer, layout, selected, sink_tokens):
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    plan = selected_read_plan(layout.ids, layout.chunks, layout.initial, selected, separator)
    mapped = selected_chunk_positions(layout, selected, plan, len(separator))
    original = [layout.ids[position] for index in selected for position in layout.chunks[index]]
    evidence = [plan.ids[position] for chunk in mapped for position in chunk]
    if original != evidence:
        raise ValueError("Selected cache tokens no longer map to the original chunk tokens")
    sink_changed = sum((u < sink_tokens) != (v < sink_tokens) for index, chunk in zip(selected, mapped)
                       for u, v in zip(layout.chunks[index], chunk))
    tokens = list(plan.ids[layout.initial:plan.document_end])
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    canonical = tokenizer.encode(text, add_special_tokens=False)
    recoded_text = tokenizer.decode(canonical, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if text != recoded_text:
        raise ValueError("Selected evidence text cannot be round-tripped by the local tokenizer")
    return dict(exact_selected_token_mapping=True, sink_membership_changed=sink_changed,
                selected_source_tokens=len(original), evidence_tokens=len(tokens),
                canonical_tokens=len(canonical), canonical_identical=tokens == canonical,
                canonical_length_delta=len(canonical)-len(tokens),
                replacement_characters=text.count("\ufffd"), separators_added=plan.separators_added)


def summarize(rows):
    if not rows:
        raise ValueError("At least one evidence plan is required")
    return dict(cases=len(rows), exact_selected_token_mapping=sum(r["exact_selected_token_mapping"] for r in rows),
        sink_membership_changed_cases=sum(r["sink_membership_changed"] > 0 for r in rows),
        canonical_identical_cases=sum(r["canonical_identical"] for r in rows),
        replacement_character_cases=sum(r["replacement_characters"] > 0 for r in rows),
        total_selected_source_tokens=sum(r["selected_source_tokens"] for r in rows),
        total_evidence_tokens=sum(r["evidence_tokens"] for r in rows),
        total_canonical_tokens=sum(r["canonical_tokens"] for r in rows),
        absolute_token_count_change=sum(abs(r["canonical_length_delta"]) for r in rows),
        total_separators_added=sum(r["separators_added"] for r in rows),
        matches_saved_full_prompt_token_counts=sum(r.get("matches_saved_full_prompt_token_counts", False) for r in rows))


def audit(workspace):
    diagnostic = workspace / "query_cartridges_gain_transfer_20261104"
    manifest = json.loads((diagnostic / "manifest.json").read_text())
    cases = manifest["case_ids"]
    if (len(cases) != 40 or len(set(cases)) != 40 or manifest["models"] != ["qwen", "gpt"] or
        manifest["reference_answers_used"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("Expected the fixed 40-case unlabeled diagnostic")
    profile = json.loads((workspace / "query_cartridges_value_lr_20261104/olmo_profile.json").read_text())
    protocols = json.loads((workspace / "query_cartridges_linked_selection_20261104/manifest.json").read_text())
    models = {}
    for name in ("qwen", "gpt"):
        spec = json.loads(Path(protocols["model_protocols"][name]).read_text())
        tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], local_files_only=True)
        if name == "gpt":
            tokenizer = FinalChannelTokenizer(tokenizer, "2026-11-03")
        engine = SimpleNamespace(tokenizer=tokenizer, config=replace(TTTConfig(**profile), max_context=131072))
        index = {entry["key"]: entry for entry in json.loads(Path(spec["index"]).read_text())}
        selector = LinkedFrozenSelector(spec["shared_scores"])
        rows = []
        with open(spec["source_cases"], "rb") as source:
            for case_key in cases:
                entry = index[case_key]
                source.seek(entry["offset"])
                case = json.loads(source.read(entry["bytes"]))
                if case["key"] != case_key or set(case) & {"answer", "answers", "references", "gold", "answer_spans"}:
                    raise ValueError("The source case unexpectedly contains reference answers")
                request = Request(document=case["document"], question=case["query"], prompt=case["prompt"],
                                  answer_prefix=case.get("answer_prefix", ""), max_new_tokens=case["max_new_tokens"])
                selector.bind_case(case_key)
                prepared = selector.prepare(engine, request)
                if (prepared.metadata["reference_answers_used"] or prepared.metadata["online_api_calls"] != 0 or
                    not prepared.metadata["complete_original_document"]):
                    raise ValueError("Selection is not the full-input, cached-score protocol")
                recorded = json.loads((diagnostic / "cases" / name / (case_key + ".json")).read_text())["selection"]
                shared_fields = ("source_case_key", "original_prompt_tokens", "working_prompt_tokens", "total_chunks")
                if any(prepared.metadata[key] != recorded[key] for key in shared_fields):
                    raise ValueError("The local tokenizer and saved full-prompt metadata disagree")
                row = inspect_plan(tokenizer, prepared.layout, prepared.selected, engine.config.sink_tokens)
                row["matches_saved_full_prompt_token_counts"] = True
                rows.append(row)
        models[name] = summarize(rows)
    return dict(scope="40 fixed source-only development cases per model", selection_policy="decision_entity_link_v1",
        independently_selected_token_plan=True, model_weights_loaded=False, model_calls=0, online_api_calls=0,
        reference_answers_used=False, predictions_or_accuracy_not_measured=True, models=models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(args.output)
    for name, data in result["models"].items():
        print(name, "cases", data["cases"], "exact maps", data["exact_selected_token_mapping"],
              "canonical ids identical", data["canonical_identical_cases"],
              "sink changes", data["sink_membership_changed_cases"])


if __name__ == "__main__":
    main()
