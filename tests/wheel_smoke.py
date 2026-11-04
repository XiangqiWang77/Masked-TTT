"""Run the packaged public selector and KV-only path on tiny Qwen/GPT models.

Invoke with a built wheel on PYTHONPATH, from outside the source checkout.
"""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import masked_ttt
from masked_ttt import FrozenJSONSelector, MaskedTTT, Request
from masked_ttt.kv_only import kv_only_cartridge
from masked_ttt.linked_selection import LinkedFrozenSelector
from test_materialization_equivalence import gpt_gqa_engine
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
             decision_probabilities={"yes": {1:.9,2:.005,3:.8,6:.85}.get(i,.01)})
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
        source_record = (json.dumps(dict(key="case", document=document, query=question)) + "\n").encode()
        source_path, index_path = root / "source.jsonl", root / "source_index.json"
        source_path.write_bytes(source_record)
        index_path.write_text(json.dumps([dict(key="case", offset=0, bytes=len(source_record))]))
        for name, factory in (("qwen", make_engine), ("gpt-gqa", gpt_gqa_engine)):
            for policy, selector_type, selected in (
                ("score-ranked", FrozenJSONSelector, [0,1,3,6]),
                ("entity-linked", LinkedFrozenSelector, [1,2,3,6]),
            ):
                engine = factory()
                engine.config = replace(
                    engine.config, keep_fraction=.1, self_question_count=0, guard_enabled=False,
                    selected_transfer_weight=0., release_full_before_cartridge=True,
                    release_selected_source_before_decode=True,
                )
                selector = selector_type(root, source_index=index_path, source_cases=source_path)
                selector.bind_case("case")
                request = Request(document=document, question=question, max_new_tokens=3)
                with kv_only_cartridge():
                    result = MaskedTTT(engine, selector).answer(request)
                memory, training = result["standalone_cartridge"]["memory"], result["training"]
                checks = dict(selection=result["selection"]["selected_chunks"] == selected,
                    source_matches=result["selection"]["source_document_matches_index"],
                    teacher_kl=training["objective"] == "on-policy KL(teacher || student)",
                    no_extra_loss=training["selected_transfer_weight"] == 0,
                    gain_gradient=any(step["gain_gradient_norm"] > 0 for step in training["steps"]),
                    value_gradient=any(step["value_gradient_norm"] > 0 for step in training["steps"]),
                    frozen_model=result["backbone_unchanged"], frozen_cache=result["original_document_KV_unchanged"],
                    materialized=memory["kv_only_materialized"], no_adapter=memory["cartridge_adapter_bytes"] == 0,
                    no_source=memory["additional_source_kv_bytes_before_decode"] == 0,
                    no_training=memory["additional_training_adapter_bytes_before_decode"] == 0,
                    no_full_cache=(result["memory"]["full_cache_release_before_standalone"]
                                   ["full_cache_tensors_live_before_standalone"] == 0))
                failed = [key for key, passed in checks.items() if not passed]
                if failed:
                    raise AssertionError("The packaged " + policy + " cartridge failed on " + name + ": " +
                                         ", ".join(failed) + "; selected=" + str(result["selection"]["selected_chunks"]))
                print(name + " / " + policy + ": selected chunks, SD and independent KV-only decode passed")


if __name__ == "__main__":
    main()
