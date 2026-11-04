from types import SimpleNamespace
from dataclasses import replace
import json
import torch

import pytest

from experiments.self_study_prompt.probe import candidate_cases,compact_stats,variant_request
import experiments.self_study_prompt.probe as probe
import experiments.babilong_linked_selection.proper_where_selector as selector_module
from experiments.self_study_prompt.enable_in_job import JOB,MODULES,REPO,enable
from experiments.self_study_prompt.report import summarize
from masked_ttt import Request
from masked_ttt._layout import selected_read_plan
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from masked_ttt.self_study_runtime import suffixes
from test_standalone import make_engine


def test_reframing_preserves_source_and_selected_kv_prefix():
    engine=make_engine()
    document="John picked up the apple. "*12
    question="How many objects is John carrying?"
    request=Request(document=document,question=question,
        prompt="Read the context.\n<context>\n"+document+"\n</context>\n\nQuestion: "+question,
        max_new_tokens=4)
    layout=prepare_layout(engine,request)
    selected=[0]
    separator=engine.tokenizer.encode("\n\n",add_special_tokens=False)
    plan=selected_read_plan(layout.ids,layout.chunks,layout.initial,selected,separator)
    teacher=SimpleNamespace(document_cache=None,document_end=plan.document_end,
                            document_ids=tuple(plan.ids[:plan.document_end]))
    variant=variant_request(request)
    assert variant.document==document and variant.question==question
    assert "</context>" in variant.prompt and "do not answer it" in variant.prompt
    student,focused=suffixes(engine,layout,selected,teacher,variant)
    assert len(student)>0 and len(focused.question_tokens)>0


def test_probe_uses_five_sources_at_two_lengths_only():
    cases=[f"babilong-{length}-qa7-{source}" for source in range(5) for length in (8,32)]
    manifest=dict(development_case_ids=cases+["babilong-16-qa1-5"])
    assert len(candidate_cases(manifest))==10
    with pytest.raises(ValueError):candidate_cases(dict(development_case_ids=cases[:-1]))


def test_probe_statistics_do_not_expose_generated_questions():
    info=dict(generated_tokens=3,parsed=0,verified=0,rejections={"invalid_json":1},
              reference_answers_used=False,student_answer_hint_used=False,
              checks=[dict(question="private",answer="private")])
    saved=compact_stats(info)
    assert saved["rejections"]=={"invalid_json":1} and "checks" not in saved


def test_optional_hook_is_idempotent_and_does_not_change_frozen_runtime(tmp_path):
    root=tmp_path/JOB;frozen=root/"runtime/repo";frozen.mkdir(parents=True)
    (root/"submission.json").write_text(json.dumps(dict(job_id="123",model="qwen",gpus=1)))
    (root/"manifest.json").write_text(json.dumps(dict(models=["qwen"],selection_policy="decision_proper_where_v1",
        reference_answers_used_in_inference=False)))
    driver=root/"runtime/rollout_driver.py"
    text="precision=ROOT/'disabled_optional_probe'\ndriver.run_if_needed(engine,model,identity)\n"
    driver.write_text(text)
    for name in MODULES:
        dest=frozen/"src/masked_ttt"/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((REPO/"src/masked_ttt"/name).read_bytes())
    for name in ("experiments/babilong_linked_selection/proper_where_selector.py",
                 "experiments/babilong_linked_selection/question_overlap_selector.py"):
        dest=frozen/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/name).read_bytes())
    units=tmp_path/"systemd"
    value=enable(tmp_path,units)
    assert value["enabled"] and value["new_gpu_jobs"]==0 and not value["training_runtime_changed"]
    assert enable(tmp_path,units)==value
    assert driver.read_text()==text and (root/"disabled_optional_probe/ready.json").is_file()
    assert (units/"query-cartridges-proper-where-dev.service.d/self-study-probe.conf").is_file()


def test_probe_summary_compares_verified_exercises_without_label_access():
    a=dict(generated_tokens=2,parsed=0,verified=0,rejections={"invalid_json":1},
           reference_answers_used=False,student_answer_hint_used=False)
    b=dict(generated_tokens=50,parsed=1,verified=1,rejections={},
           reference_answers_used=False,student_answer_hint_used=False)
    value=dict(model="qwen",cases=[str(i) for i in range(10)],rows={"1":dict(base=a,variant=b)},
        errors=[],complete=False,reference_answers_used=False,online_api_calls=0,training_runtime_changed=False)
    output=summarize(value)
    assert output["n"]==1 and output["more_verified"]==1 and output["base"]["invalid_json"]==1
    value["rows"]["1"]["variant"]["reference_answers_used"]=True
    with pytest.raises(ValueError):summarize(value)


def test_hook_compares_both_prompts_without_mutating_model_or_rng(tmp_path,monkeypatch):
    root=tmp_path/"job";here=root/"disabled_optional_probe";here.mkdir(parents=True)
    key="babilong-8-qa7-1";question="How many objects is John carrying?"
    document="John picked up the apple. "*12
    case=dict(key=key,document=document,query=question,
              prompt="<context>\n"+document+"\n</context>\nQuestion: "+question,max_new_tokens=4)
    encoded=(json.dumps(case)+"\n").encode();source=root/"cases.jsonl";source.write_bytes(encoded)
    index=root/"index.json";index.write_text(json.dumps([dict(key=key,offset=0,bytes=len(encoded))]))
    protocol=root/"protocol.json";protocol.write_text(json.dumps(dict(index=str(index),source_cases=str(source),
                                                                       shared_scores=str(root/"unused"))))
    manifest=dict(models=["qwen"],selection_policy="decision_proper_where_v1",
        reference_answers_used_in_inference=False,online_api_calls=0,model_protocols=dict(qwen=str(protocol)))
    (root/"manifest.json").write_text(json.dumps(manifest))
    class Selector:
        def __init__(self,_path):pass
        def bind_case(self,_key):pass
        def prepare(self,engine,request):
            return PreparedRequest(request,prepare_layout(engine,request),[0],dict())
    monkeypatch.setattr(probe,"HERE",here)
    monkeypatch.setattr(probe,"candidate_cases",lambda _manifest:[key])
    monkeypatch.setattr(selector_module,"ProperWhereSelector",Selector)
    monkeypatch.setattr(probe,"prefill_document",lambda *_args:object())
    seen=[]
    def fake_study(_engine,_reader,_teacher,request,_layout,_selected):
        variant="do not answer it" in request.prompt;seen.append(variant)
        info=dict(generated_tokens=50 if variant else 2,parsed=int(variant),verified=int(variant),
                  rejections={} if variant else {"invalid_json":1},reference_answers_used=False,
                  student_answer_hint_used=False,checks=[dict(answer="discarded")])
        return [],info
    monkeypatch.setattr(probe,"generate_self_study",fake_study)
    engine=make_engine();rng=torch.get_rng_state().clone();stamp=engine.weight_stamp()
    assert probe.run_if_needed(engine,"qwen",dict(gpu="CPU test"))=="complete"
    saved=json.loads((here/"qwen.json").read_text())
    assert seen==[False,True] and saved["complete"]
    assert saved["rows"][key]["variant"]["verified"]==1 and "checks" not in saved["rows"][key]["variant"]
    assert torch.equal(rng,torch.get_rng_state());engine.assert_frozen(stamp)


def test_hook_executes_real_selected_prefill_and_self_study_on_small_model(tmp_path,monkeypatch):
    root=tmp_path/"job";here=root/"disabled_optional_probe";here.mkdir(parents=True)
    key="babilong-8-qa7-1";question="How many objects is John carrying?"
    document="John picked up the apple. "*6
    case=dict(key=key,document=document,query=question,
              prompt="<context>\n"+document+"\n</context>\nQuestion: "+question,max_new_tokens=4)
    encoded=(json.dumps(case)+"\n").encode();source=root/"cases.jsonl";source.write_bytes(encoded)
    index=root/"index.json";index.write_text(json.dumps([dict(key=key,offset=0,bytes=len(encoded))]))
    protocol=root/"protocol.json";protocol.write_text(json.dumps(dict(index=str(index),source_cases=str(source),
                                                                       shared_scores=str(root/"unused"))))
    (root/"manifest.json").write_text(json.dumps(dict(models=["qwen"],selection_policy="decision_proper_where_v1",
        reference_answers_used_in_inference=False,online_api_calls=0,model_protocols=dict(qwen=str(protocol)))))
    class Selector:
        def __init__(self,_path):pass
        def bind_case(self,_key):pass
        def prepare(self,engine,request):
            return PreparedRequest(request,prepare_layout(engine,request),[0],dict())
    monkeypatch.setattr(probe,"HERE",here)
    monkeypatch.setattr(probe,"candidate_cases",lambda _manifest:[key])
    monkeypatch.setattr(selector_module,"ProperWhereSelector",Selector)
    engine=make_engine();engine.eos_ids={-1}
    engine.config=replace(engine.config,self_question_count=1,self_question_max_tokens=4,
                          self_question_verify_tokens=2,prefill_chunk_tokens=64)
    rng=torch.get_rng_state().clone();stamp=engine.weight_stamp()
    assert probe.run_if_needed(engine,"qwen",dict(gpu="CPU test"))=="complete"
    result=json.loads((here/"qwen.json").read_text())
    assert not result["errors"] and result["rows"][key]["base"]["generated_tokens"]==4
    assert result["rows"][key]["variant"]["generated_tokens"]==4
    assert torch.equal(rng,torch.get_rng_state());engine.assert_frozen(stamp)
