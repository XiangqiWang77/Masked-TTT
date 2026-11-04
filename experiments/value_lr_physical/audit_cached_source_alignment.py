#!/usr/bin/env python3
"""Validate indexed source records and cached-score structure without hashing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from masked_ttt.frozen_scores import FrozenJSONSelector, load_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    protocols = {name: json.loads(Path(path).read_text()) for name, path in manifest["model_protocols"].items()}
    if set(protocols) != {"qwen", "gpt"}:
        raise ValueError("Both target model protocols are required")
    keys = ("index", "source_cases", "shared_scores")
    if any(Path(protocols["qwen"][key]).resolve() != Path(protocols["gpt"][key]).resolve() for key in keys):
        raise ValueError("The model protocols do not share the same indexed source and scores")
    protocol = protocols["qwen"]
    index = json.loads(Path(protocol["index"]).read_text())
    if len(index) != 1000 or len({row["key"] for row in index}) != 1000:
        raise ValueError("Expected 1,000 unique indexed source cases")
    selector = FrozenJSONSelector(protocol["shared_scores"], source_index=protocol["index"],
                                  source_cases=protocol["source_cases"])
    buckets, chunks = Counter(), []
    with open(protocol["source_cases"], "rb") as stream:
        for row in sorted(index, key=lambda x: x["offset"]):
            stream.seek(row["offset"])
            case = json.loads(stream.read(row["bytes"]))
            key, document, question = case["key"], case["document"], case["query"]
            if key != row["key"]:
                raise ValueError("Indexed source key changed")
            selector.bind_case(key)
            selector.indexed_source.verify(key, document, question)
            data = load_scores(selector.directory, document, question, key, selector.by_case)
            buckets[str(row["length_bucket"])] += 1
            chunks.append(len(data["records"]))
    result = dict(cases=len(index), models=["qwen", "gpt"], shared_index_source_and_scores=True,
                  query_and_chunk_spans_checked=True, indexed_record_reloaded=True,
                  online_api_calls=0, hash_calculations=0, reference_answers_used=False,
                  length_bucket_counts=dict(sorted(buckets.items())),
                  min_chunks=min(chunks), max_chunks=max(chunks),
                  scope="Indexed records and score key/question/span structure; score provenance is supplied by the saved files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
