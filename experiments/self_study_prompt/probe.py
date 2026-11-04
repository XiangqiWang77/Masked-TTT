#!/usr/bin/env python3
"""Compare two self-study question framings on one already-loaded Qwen GPU.

This optional probe does not change the submitted job's model, optimizer,
objective, predictions, or selector. It reuses a selected-only teacher KV to
generate independently verified local study exercises for at most ten saved
development questions. It reads neither evaluation references nor answers.
"""
from contextlib import nullcontext
from copy import copy
import fcntl
import gc
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import torch
from masked_ttt import Request
from masked_ttt._layout import selected_read_plan
from masked_ttt.model import Budget
from masked_ttt.request import prepare_layout
from masked_ttt.runtime import private_module_tree
from masked_ttt.self_study_runtime import generate_self_study
from masked_ttt.teacher import prefill_document

HERE = Path(__file__).resolve().parent


def variant_request(request):
    """Remove the competing direct-answer label while preserving the document prefix."""
    start = request.prompt.find(request.document)
    if start < 0 or request.prompt.find(request.document,start+1) >= 0:
        raise ValueError("The original source must appear exactly once")
    end = start + len(request.document)
    suffix = request.prompt[end:]
    # Preserve whitespace and an optional closing context tag verbatim. The
    # prefix check in suffixes() rejects token-boundary changes.
    closing = re.match(r"\s*(?:</[A-Za-z][A-Za-z0-9:_-]*>\s*)*",suffix).group()
    prompt = (request.prompt[:end] + closing +
              "Original question for identifying study facts only; do not answer it: " + request.question)
    return Request(document=request.document,question=request.question,prompt=prompt,
                   answer_prefix="",max_new_tokens=request.max_new_tokens)


def candidate_cases(manifest):
    cases = [case for case in manifest["development_case_ids"]
             if case.startswith("babilong-8-qa7-") or case.startswith("babilong-32-qa7-")]
    sources = {case.split("-",2)[2] for case in cases}
    if len(cases) != 10 or len(sources) != 5 or len(set(cases)) != 10:
        raise ValueError("The fixed QA7 source-only probe set changed")
    return sorted(cases)


def compact_stats(info):
    return dict(generated_tokens=info["generated_tokens"],parsed=info["parsed"],
                verified=info["verified"],rejections=info["rejections"],
                reference_answers_used=info["reference_answers_used"],
                student_answer_hint_used=info["student_answer_hint_used"])


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def run_if_needed(engine, model, identity):
    if model != "qwen":return "not_applicable"
    from experiments.babilong_linked_selection.proper_where_selector import ProperWhereSelector
    from masked_ttt.linked_selection import LinkedFrozenSelector
    root = HERE.parent
    manifest = json.loads((root/"manifest.json").read_text())
    if (manifest["models"] != ["qwen"] or manifest["selection_policy"] not in
        ("decision_proper_where_v1", "decision_entity_link_v1") or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0):
        raise ValueError("The probe does not match the frozen Qwen job")
    cases = candidate_cases(manifest)
    with (HERE/"probe.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        output = HERE/"qwen.json"
        result = (json.loads(output.read_text()) if output.exists() else
                  dict(model="qwen",cases=cases,rows={},errors=[],complete=False,
                       reference_answers_used=False,online_api_calls=0,training_runtime_changed=False))
        if result["cases"] != cases or result["model"] != "qwen":
            raise ValueError("The saved source-only probe changed")
        if result["complete"]:return "already_complete"
        spec = json.loads(Path(manifest["model_protocols"]["qwen"]).read_text())
        entries = {entry["key"]:entry for entry in json.loads(Path(spec["index"]).read_text())}
        selector_type = (ProperWhereSelector if manifest["selection_policy"] == "decision_proper_where_v1"
                         else LinkedFrozenSelector)
        selector = selector_type(spec["shared_scores"])
        local = copy(engine)
        # The driver invokes optional probes only with >15 minutes remaining.
        # Bound this diagnostic to six minutes for the whole allocation.
        deadline = time.monotonic()+360
        local.budget = Budget(360)
        reader = SimpleNamespace(model=private_module_tree(local.model.model),lm_head=local.model.lm_head)
        stamp = local.weight_stamp()
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if local.device.type == "cuda" else None
        scope = getattr(local.tokenizer,"request_scope",None)
        try:
            with (scope() if callable(scope) else nullcontext()),open(spec["source_cases"],"rb") as source:
                for key in cases:
                    if key in result["rows"]:continue
                    if sum(item["case"] == key for item in result["errors"]) >= 2:continue
                    if time.monotonic() >= deadline-20:break
                    try:
                        local.budget.check()
                        entry=entries[key];source.seek(entry["offset"])
                        case=json.loads(source.read(entry["bytes"]))
                        if case["key"] != key or set(case)&{"answer","answers","references","gold","answer_spans"}:
                            raise ValueError("A source case contains reference labels")
                        request=Request(document=case["document"],question=case["query"],prompt=case["prompt"],
                                        answer_prefix=case.get("answer_prefix",""),max_new_tokens=case["max_new_tokens"])
                        selector.bind_case(key);prepared=selector.prepare(local,request)
                        working,layout,selected=prepared.request,prepared.layout,prepared.selected
                        instruction=local.config.teacher_response_instruction
                        teacher_request=Request(document=working.document,question=working.question,
                            prompt=working.prompt+"\n\nResponse requirements:\n"+instruction,
                            answer_prefix=working.answer_prefix,max_new_tokens=working.max_new_tokens)
                        teacher_layout=prepare_layout(local,teacher_request)
                        if teacher_layout.ids[:layout.document_end] != layout.ids[:layout.document_end]:
                            raise ValueError("The Teacher document prefix changed")
                        separator=local.tokenizer.encode("\n\n",add_special_tokens=False)
                        plan=selected_read_plan(teacher_layout.ids,layout.chunks,layout.initial,selected,separator)
                        cache=prefill_document(local,reader,plan.ids,plan.initial_prefix_tokens,plan.chunk_boundaries)
                        teacher=SimpleNamespace(document_cache=cache,document_end=plan.document_end,
                                                document_ids=tuple(plan.ids[:plan.document_end]))
                        base_paths,base=generate_self_study(local,reader,teacher,working,layout,selected)
                        del base_paths
                        variant=variant_request(working)
                        candidate_paths,candidate=generate_self_study(local,reader,teacher,variant,layout,selected)
                        del candidate_paths,teacher,cache
                        result["rows"][key]=dict(base=compact_stats(base),variant=compact_stats(candidate))
                        atomic(output,result)
                    except TimeoutError:
                        break
                    except Exception as exc:
                        result["errors"].append(dict(case=key,error_type=type(exc).__name__))
                        atomic(output,result)
                        break
                    finally:
                        gc.collect()
                        if local.device.type == "cuda":
                            with local.device_context():torch.cuda.empty_cache()
            local.assert_frozen(stamp)
            result.update(complete=len(result["rows"]) == len(cases),identity=identity,updated=time.time())
            atomic(output,result)
        finally:
            torch.set_rng_state(rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        return "complete" if result["complete"] else "partial"
