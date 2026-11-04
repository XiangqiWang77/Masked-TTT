"""Estimate KV-only artifact size from fixed completed-run accounting records."""
import argparse
import json
from pathlib import Path
import statistics


def read_row(case, result):
    full = result["memory"]["full_document_kv_bytes"]
    memory = result["standalone_cartridge"]["memory"]
    artifact = memory["standalone_cartridge_bytes"]
    kv = memory["materialized_selected_kv_bytes"]
    adapter = memory["cartridge_adapter_bytes"]
    if (any(type(value) is not int for value in (full, artifact, kv, adapter)) or
        full <= 0 or artifact <= 0 or kv <= 0 or adapter < 0 or kv > artifact):
        raise ValueError("Invalid completed-case storage accounting")
    return dict(
        case=case, full_bytes=full, artifact_before_bytes=artifact,
        materialized_kv_bytes=kv, reported_adapter_bytes=adapter,
        estimated_saved_bytes=artifact-kv,
        artifact_before_percent=100*artifact/full,
        kv_only_percent=100*kv/full,
    )


def audit(snapshot, root):
    rows, unavailable = [], []
    for item in snapshot["rows"]:
        case = item["case"]
        path = Path(root)/"cases"/case/"weight_0.00.json"
        try:
            result = json.loads(path.read_text())
            if result["key"] != case:
                raise ValueError("Case mismatch")
            rows.append(read_row(case, result))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            unavailable.append(dict(case=case, error=type(exc).__name__))
    def median(name):
        return statistics.median(row[name] for row in rows) if rows else None
    return dict(
        model=snapshot["model"], snapshot_time=snapshot["updated"], n=len(rows),
        expected=len(snapshot["rows"]), unavailable=unavailable,
        median_full_bytes=median("full_bytes"),
        median_artifact_before_bytes=median("artifact_before_bytes"),
        median_materialized_kv_bytes=median("materialized_kv_bytes"),
        median_estimated_saved_bytes=median("estimated_saved_bytes"),
        median_artifact_before_percent=median("artifact_before_percent"),
        median_kv_only_percent=median("kv_only_percent"),
        estimate_only=True, excludes_model_and_training_peak=True,
        new_model_calls=0, online_api_calls=0, rows=rows,
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument("results_root",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=audit(json.loads(args.snapshot.read_text()),args.results_root)
    temporary=args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result,indent=2)+"\n")
    temporary.replace(args.output.with_suffix(".json"))


if __name__=="__main__":
    main()
