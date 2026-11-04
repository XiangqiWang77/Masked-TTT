import json

import pytest

from experiments.kv_only_cartridge.audit_storage import audit, read_row


def _result():
    return dict(key="case", memory=dict(full_document_kv_bytes=1000),
        standalone_cartridge=dict(memory=dict(
            standalone_cartridge_bytes=200, materialized_selected_kv_bytes=150,
            cartridge_adapter_bytes=50)))


def test_storage_estimate_uses_materialized_kv_and_excludes_model():
    result=read_row("case",_result())
    assert result["artifact_before_percent"]==20
    assert result["kv_only_percent"]==15
    assert result["estimated_saved_bytes"]==50
    bad=_result()
    bad["standalone_cartridge"]["memory"]["materialized_selected_kv_bytes"]=201
    with pytest.raises(ValueError):read_row("case",bad)


def test_audit_uses_fixed_snapshot_and_records_missing_cases(tmp_path):
    path=tmp_path/"cases/case/weight_0.00.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_result()))
    snapshot=dict(model="test",updated=1,rows=[dict(case="case"),dict(case="missing")])
    result=audit(snapshot,tmp_path)
    assert result["n"]==1 and result["expected"]==2 and len(result["unavailable"])==1
    assert result["median_artifact_before_percent"]==20
    assert result["median_kv_only_percent"]==15
    assert result["estimate_only"] and result["excludes_model_and_training_peak"]
