"""Run the packaged public selector and KV-only path on tiny Qwen/GPT models.

Invoke with a built wheel on PYTHONPATH, from outside the source checkout.
"""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import masked_ttt
from masked_ttt import MaskedTTT, Request
from masked_ttt.kv_only import kv_only_cartridge
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def main():
    if ".whl/" not in str(masked_ttt.__file__):
        raise AssertionError("Smoke test must import from the built wheel")
    chunks = ["Other."] * 40
    chunks[1] = "Daniel grabbed the milk."
    chunks[2] = "Daniel moved to the garden."
    chunks[6] = "Daniel dropped the milk."
    document = "\n".join(chunks)
    question = "Where is the milk?"
    spans, cursor = [], 0
    for text in chunks:
        start = document.index(text, cursor)
        end = start + len(text)
        spans.append([start, end])
        cursor = end
    records = [
        dict(chunk=i, utility_output_valid=True,
             decision_probabilities={"yes": {1:.9,2:.05,3:.8,6:.85}.get(i,.01)})
        for i in range(len(chunks))
    ]
    with TemporaryDirectory(prefix="masked-ttt-smoke-") as directory:
        root = Path(directory)
        (root / "scores").mkdir()
        (root / "case_to_score_file.json").write_text(json.dumps({"case":"case.json"}))
        (root / "scores/case.json").write_text(json.dumps(dict(
            schema="masked-ttt-source-scores-v1", complete=True, case_key="case",
            question=question, reference_answers_used=False, source_spans=spans,
            canonical_tokenizer="test", scoring_seconds=0., estimated_cost_usd=0., records=records,
        )))
        for name, factory in (("qwen", make_engine), ("gpt", gpt_engine)):
            engine = factory()
            engine.config = replace(
                engine.config, keep_fraction=.1, self_question_count=0, guard_enabled=False,
                selected_transfer_weight=0., release_full_before_cartridge=True,
                release_selected_source_before_decode=True,
            )
            selector = LinkedFrozenSelector(root)
            selector.bind_case("case")
            request = Request(document=document, question=question, max_new_tokens=3)
            with kv_only_cartridge():
                result = MaskedTTT(engine, selector).answer(request)
            memory = result["standalone_cartridge"]["memory"]
            if (result["selection"]["selected_chunks"] != [1,2,3,6] or
                not memory["kv_only_materialized"] or memory["cartridge_adapter_bytes"] != 0 or
                memory["additional_source_kv_bytes_before_decode"] != 0 or
                memory["additional_training_adapter_bytes_before_decode"] != 0 or
                result["memory"]["full_cache_release_before_standalone"]["full_cache_tensors_live_before_standalone"] != 0):
                raise AssertionError("The packaged standalone cartridge failed on " + name)
            print(name + ": selected chunks, SD and independent KV-only decode passed")


if __name__ == "__main__":
    main()
