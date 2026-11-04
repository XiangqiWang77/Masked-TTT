from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.monitoring import due_check


def setup_schedule(tmp_path,now):
    directory=tmp_path/"monitor";directory.mkdir()
    path=directory/"cartridge_user_interval_20261104.json"
    data=dict(rollout_value_qwen_job_id="123",canonical_pair_job_id="456",
        rollout_value_qwen_next_check_not_before_utc=(now+timedelta(minutes=1)).isoformat(),
        canonical_pair_next_check_not_before_utc=now.isoformat())
    path.write_text(json.dumps(data))
    return path,data


def test_only_due_job_is_queried_or_read_and_deadline_advances(tmp_path):
    now=datetime(2026,11,4,14,30,tzinfo=timezone.utc)
    path,_=setup_schedule(tmp_path,now)
    calls=[]
    def query(job,_workspace):
        calls.append(("query",job));return dict(state="RUNNING",reason="node",terminal=False)
    def summary(_workspace,name):
        calls.append(("summary",name));return dict(models={"qwen":dict(n=21)})
    def probe(_workspace,name):
        calls.append(("probe",name));return None
    observed=due_check.run(tmp_path,now=now,query=query,summary=summary,probe=probe)
    assert [x["job_id"] for x in observed]==["456"]
    assert calls==[("query","456"),("summary","canonical_pair"),("probe","canonical_pair")]
    after=json.loads(path.read_text())
    assert after["canonical_pair_next_check_not_before_utc"]==(now+timedelta(hours=1)).isoformat()
    assert after["rollout_value_qwen_next_check_not_before_utc"]==(now+timedelta(minutes=1)).isoformat()
    assert due_check.run(tmp_path,now=now,query=query,summary=summary,probe=probe)==[]
    assert len(calls)==3


def test_unknown_state_waits_and_confirmed_terminal_stops_polling(tmp_path):
    now=datetime(2026,11,4,14,30,tzinfo=timezone.utc);path,data=setup_schedule(tmp_path,now)
    data["rollout_value_qwen_next_check_not_before_utc"]=now.isoformat()
    path.write_text(json.dumps(data))
    def query(job,_workspace):
        return dict(state="UNKNOWN" if job=="123" else "COMPLETED",terminal=job=="456")
    result=due_check.run(tmp_path,now=now,query=query,summary=lambda *_:None,probe=lambda *_:None)
    assert len(result)==2
    state=json.loads(path.read_text())
    assert state["rollout_value_qwen_next_check_not_before_utc"]==(now+timedelta(hours=1)).isoformat()
    # A completed parent can still be waiting for its hourly promotion logic.
    assert state["canonical_pair_next_check_not_before_utc"]==(now+timedelta(hours=1)).isoformat()
    root=tmp_path/due_check.SPECS["canonical_pair"];root.mkdir()
    (root/"monitor_status.json").write_text(json.dumps(dict(state="not_eligible")))
    due_check.run(tmp_path,now=now+timedelta(hours=1),query=query,summary=lambda *_:None,probe=lambda *_:None)
    state=json.loads(path.read_text())
    assert state["canonical_pair_next_check_not_before_utc"] is None


def test_promoted_holdout_is_registered_without_querying_it(tmp_path):
    now=datetime(2026,11,4,14,30,tzinfo=timezone.utc);path,data=setup_schedule(tmp_path,now)
    data["rollout_value_qwen_next_check_not_before_utc"]=now.isoformat()
    data["canonical_pair_next_check_not_before_utc"]=None
    path.write_text(json.dumps(data))
    parent=tmp_path/due_check.SPECS["rollout_value_qwen"];parent.mkdir()
    (parent/"monitor_status.json").write_text(json.dumps(dict(state="submitted")))
    holdout=tmp_path/due_check.SPECS["rollout_value_qwen_holdout"];holdout.mkdir()
    (holdout/"submission.json").write_text(json.dumps(dict(job_id="987",created=now.timestamp())))
    queried=[]
    def query(job,_workspace):
        queried.append(job);return dict(state="COMPLETED",terminal=True)
    observed=due_check.run(tmp_path,now=now,query=query,summary=lambda *_:None,probe=lambda *_:None)
    assert queried==["123"] and observed[0]["new_holdouts"]==[
        dict(name="rollout_value_qwen_holdout",job_id="987")]
    state=json.loads(path.read_text())
    assert state["rollout_value_qwen_next_check_not_before_utc"] is None
    assert state["rollout_value_qwen_holdout_job_id"]=="987"
    assert state["rollout_value_qwen_holdout_next_check_not_before_utc"]==(now+timedelta(hours=1)).isoformat()


def test_summary_extracts_only_aggregate_values(tmp_path):
    root=tmp_path/due_check.SPECS["canonical_pair"];root.mkdir()
    (root/"latest.json").write_text(json.dumps(dict(metric="answer_focused_lexical_v2",expected_per_model=200,
        errors=[],invalid=[],models={"qwen":dict(overall=dict(n=2,accuracy={"canonical_cartridge":50}),
        rows=[dict(answer="private")])})))
    out=due_check.read_summary(tmp_path,"canonical_pair")
    assert out["models"]["qwen"]["n"]==2 and out["errors"]==0
    assert "private" not in json.dumps(out) and "rows" not in json.dumps(out)


def test_scheduler_unavailable_is_not_a_terminal_job(tmp_path,monkeypatch):
    def fail(*_args,**_kwargs):raise ValueError("synthetic")
    monkeypatch.setattr(due_check.subprocess,"run",fail)
    with pytest.raises(ValueError):
        # subprocess.TimeoutExpired is the supported timeout; an unrelated
        # programming exception should remain visible.
        due_check.query_scheduler("123",tmp_path)
    def timeout(*_args,**_kwargs):
        raise due_check.subprocess.TimeoutExpired(cmd="squeue",timeout=20)
    monkeypatch.setattr(due_check.subprocess,"run",timeout)
    assert due_check.query_scheduler("123",tmp_path)["terminal"] is False
    monkeypatch.setattr(due_check.subprocess,"run",lambda *_args,**_kwargs:
                        SimpleNamespace(returncode=0,stdout="123|RUNNING|node|1:00\n"))
    assert due_check.query_scheduler("123",tmp_path)["state"]=="RUNNING"


def test_missing_queue_uses_terminal_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(due_check.subprocess,"run",lambda *_args,**_kwargs:
                        SimpleNamespace(returncode=0,stdout=""))
    missing=due_check.query_scheduler("123",tmp_path)
    assert missing["state"]=="UNKNOWN" and not missing["terminal"]
    (tmp_path/"cartridge_job_watch.py").write_text("def observed_terminal(job):\n    return None\n")
    result=due_check.query_scheduler("123",tmp_path)
    assert result["state"]=="UNKNOWN" and not result["terminal"]
    (tmp_path/"cartridge_job_watch.py").write_text("def observed_terminal(job):\n    return 'COMPLETED'\n")
    # Avoid importlib pycache freshness and verify the confirmed terminal path.
    monkeypatch.setattr(due_check.importlib.util,"module_from_spec",lambda _spec:
                        SimpleNamespace(observed_terminal=lambda _job:"COMPLETED"))
    monkeypatch.setattr(due_check.importlib.util,"spec_from_file_location",lambda *_args:
                        SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module:None)))
    result=due_check.query_scheduler("123",tmp_path)
    assert result["state"]=="COMPLETED" and result["terminal"]
