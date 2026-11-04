#!/usr/bin/env python3
"""Check the real Qwen tokenizer preserves both document-prefix views."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from experiments.babilong_linked_selection.proper_where_selector import ProperWhereSelector
from experiments.self_study_prompt.probe import candidate_cases,variant_request
from masked_ttt import Request,TTTConfig
from masked_ttt._layout import selected_read_plan
from masked_ttt.request import prepare_layout
from masked_ttt.self_study_runtime import suffixes


def check(root):
    manifest=json.loads((root/"manifest.json").read_text())
    profile=json.loads((root/"runtime/olmo_profile.json").read_text())
    spec=json.loads(Path(manifest["model_protocols"]["qwen"]).read_text())
    if (manifest["models"] != ["qwen"] or manifest["selection_policy"] != "decision_proper_where_v1" or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The real-tokenizer source protocol changed")
    tokenizer=AutoTokenizer.from_pretrained(spec["model_path"],local_files_only=True)
    engine=SimpleNamespace(tokenizer=tokenizer,config=replace(TTTConfig(**profile),max_context=131072),
                           device=torch.device("cpu"))
    index={entry["key"]:entry for entry in json.loads(Path(spec["index"]).read_text())}
    selector=ProperWhereSelector(spec["shared_scores"])
    checked=0
    with open(spec["source_cases"],"rb") as source:
        for key in candidate_cases(manifest):
            entry=index[key];source.seek(entry["offset"]);case=json.loads(source.read(entry["bytes"]))
            if case["key"] != key or set(case)&{"answer","answers","references","gold","answer_spans"}:
                raise ValueError("A source case contains evaluation labels")
            request=Request(document=case["document"],question=case["query"],prompt=case["prompt"],
                            answer_prefix=case.get("answer_prefix",""),max_new_tokens=case["max_new_tokens"])
            selector.bind_case(key);prepared=selector.prepare(engine,request)
            working,layout,selected=prepared.request,prepared.layout,prepared.selected
            teacher_request=Request(document=working.document,question=working.question,
                prompt=working.prompt+"\n\nResponse requirements:\n"+engine.config.teacher_response_instruction,
                answer_prefix=working.answer_prefix,max_new_tokens=working.max_new_tokens)
            teacher_layout=prepare_layout(engine,teacher_request)
            if teacher_layout.ids[:layout.document_end] != layout.ids[:layout.document_end]:
                raise ValueError("The Teacher changed the full document token prefix")
            separator=tokenizer.encode("\n\n",add_special_tokens=False)
            plan=selected_read_plan(teacher_layout.ids,layout.chunks,layout.initial,selected,separator)
            teacher=SimpleNamespace(document_cache=None,document_end=plan.document_end,
                                    document_ids=tuple(plan.ids[:plan.document_end]))
            for candidate in (working,variant_request(working)):
                student,focused=suffixes(engine,layout,selected,teacher,candidate)
                if not student.numel() or not focused.question_tokens.numel():
                    raise ValueError("An original or reframed question suffix is empty")
            checked+=1
    if checked!=10:raise ValueError("The fixed tokenizer check is incomplete")
    return dict(model="Qwen/Qwen3-4B-Instruct-2507",cases=checked,source_questions=5,
        context_lengths=[8192,32768],full_document_prefix_matches=True,
        selected_teacher_prefix_matches=True,original_and_reframed_suffixes_nonempty=True,
        model_weights_loaded=False,reference_answers_used=False,online_api_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();result=check(args.job_root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":main()
