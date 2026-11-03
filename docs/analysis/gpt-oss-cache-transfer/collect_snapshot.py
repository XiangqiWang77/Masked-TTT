#!/usr/bin/env python3
"""Create a label-only, reproducible snapshot from the saved GPT-OSS runs."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CROSS=Path("/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102")
RUN=CROSS/"query_cartridges_gpt_consistency_20261103"
OLD=CROSS/"query_cartridges_gpt_transfer_audit_20261103/latest.json"
PROJECT=CROSS/"query_cartridges_20261103"
DIAG=PROJECT/"posthoc_diagnostics"
sys.path[:0]=[str(DIAG),str(PROJECT)]
from practice_rank_coverage import select


def main():
    latest=json.loads((RUN/"latest.json").read_text())
    historical=json.loads(OLD.read_text())
    annotated=json.loads((DIAG/"practice_rank_coverage.json").read_text())
    coverage={r["case"]:r["top_30pct"]["status"] for r in annotated["rows"]}
    spec=json.loads((PROJECT/"runs/qwen3-4b-babilong/protocol.json").read_text())
    score_dir=Path(spec["shared_scores"])
    lookup=json.loads((score_dir/"case_to_score_file.json").read_text())
    if len(historical["rows"])!=200 or not latest["rows"]:
        raise ValueError("Require the complete 200-case audit and a nonempty matched probe")
    rows=[]; kl=[]
    for item in latest["rows"]:
        key=item["case"]
        cases={w:json.loads((RUN/"cases"/key/("weight_"+w+".json")).read_text())
               for w in ("0.00","0.25","0.75")}
        score=json.loads((score_dir/"scores"/lookup[key]).read_text())
        selected=sorted({i for group in cases["0.25"]["selection"]["selected_original_chunks"] for i in group})
        if selected!=select(score["records"],"top_30pct"):
            raise ValueError("Support annotation and current selected chunks differ: "+key)
        a=cases["0.25"]
        bits=item["0.25"]
        full_bytes=a["memory"]["full_document_kv_bytes"]
        selected_bytes=a["standalone_cartridge"]["memory"]["standalone_cartridge_bytes"]
        if full_bytes<=0 or selected_bytes<=0:raise ValueError("Invalid cache storage")
        rows.append(dict(case=key,task=item["task"],length=a["length"],coverage=coverage[key],
            full_before=bits["full_before"],full_after=bits["full_after"],
            selected_before=bits["zero"],selected_after=bits["trained"],
            teacher=bits["teacher"],full_document_kv_bytes=full_bytes,
            standalone_kv_and_adapter_bytes=selected_bytes,
            selected_tokens=a["selection"]["selected_tokens"],
            working_prompt_tokens=a["selection"]["working_prompt_tokens"]))
        for w in ("0.25","0.75"):
            steps=cases[w]["training"]["steps"]
            if len(steps)<2:continue
            first,second=steps[:2]
            same=(first["question_path"]==second["question_path"] and
                  first["selected_rollout_tokens"]==second["selected_rollout_tokens"] and
                  first["selected_target_positions"]==second["selected_target_positions"])
            if not same:continue
            before=first["selected_transfer_KL_before"]
            after=second["selected_transfer_KL_before"]
            if before<=0 or after<=0:raise ValueError("Log axes require positive KL")
            kl.append(dict(case=key,weight=float(w),before=before,after=after,
                           question_path=first["question_path"]))
    out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        metric=latest["metric"],scope="incomplete 40-case development probe; matched completed cases",
        model=latest["model"],expected=latest["expected"],arms=latest["arms"],
        rows=rows,kl_same_prefix=kl,
        coverage_audit_rows=[{k:r[k] for k in ("case","coverage","full_zero","selected_zero")}
                             for r in historical["rows"]],
        provenance=dict(new_run="query_cartridges_gpt_consistency_20261103",
            older_audit="query_cartridges_gpt_transfer_audit_20261103",
            support_annotations="query_cartridges_20261103/posthoc_diagnostics/practice_rank_coverage.json",
            note="Saved annotations and answers are used only for post-hoc scoring, never inference."))
    target=HERE/"data_snapshot.json"
    target.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print("Saved",len(rows),"matched cases and",len(kl),"same-prefix KL pairs")


if __name__=="__main__":main()
