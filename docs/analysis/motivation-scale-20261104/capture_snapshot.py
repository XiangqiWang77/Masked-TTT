#!/usr/bin/env python3
"""Freeze already-scored paired results without source text or answer labels."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CROSS=Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102')
INPUTS={'Qwen3-4B':CROSS/'query_cartridges_qwen_olmo1000_20261104',
        'GPT-OSS-20B':CROSS/'query_cartridges_gpt_olmo1000_20261103'}
SUPPORT=CROSS/'query_cartridges_20261103/posthoc_diagnostics/selected_support.json'


def main():
    if (ROOT/'data_snapshot.csv').exists() or (ROOT/'data_snapshot.metadata.json').exists():
        raise RuntimeError('Snapshot already exists; choose a new directory for another snapshot')
    evidence=json.loads(SUPPORT.read_text())
    if not evidence['annotations_used_only_posthoc'] or evidence['API_calls']!=0:
        raise ValueError('Unexpected post-hoc evidence source')
    facts={row['case']:row for row in evidence['records']}
    rows=[];sources={}
    for model,folder in INPUTS.items():
        raw=list(csv.DictReader((folder/'per_case.csv').open(newline='')))
        result=json.loads((folder/'latest.json').read_text())
        if len(raw)!=result['completed'] or {row['case'] for row in raw}!={row['case'] for row in result['rows']}:
            raise ValueError('Source report is being updated or case sets do not match')
        for item in raw:
            fact=facts[item['case']]
            if item['coverage'] not in ('complete','incomplete'):
                raise ValueError('Unknown or unmatched selected support coverage')
            low=float(fact['content_fact_cover_lower']);high=float(fact['content_fact_cover_upper'])
            if not 0<=low<=high<=1 or (item['coverage']=='complete')!=(low==high==1):
                raise ValueError('Inconsistent support coverage')
            if any(item[name] not in ('0','1') for name in ('whole','selective','cartridge')):
                raise ValueError('Expected binary saved scores')
            rows.append(dict(model=model,case=item['case'],source_question=item['case'].split('-',2)[2],
                task=item['task'],length=int(item['length']),fact_cover_lower=low,fact_cover_upper=high,
                whole=int(item['whole']),selected=int(item['selective']),cartridge=int(item['cartridge'])))
        sources[model]=dict(completed=len(raw),source_folder=str(folder),
            report_utc=datetime.fromtimestamp(result['updated'],timezone.utc).isoformat(),
            metric=result['metric'],expected=result['expected'])
    fields=list(rows[0])
    with (ROOT/'data_snapshot.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        support_source=str(SUPPORT),support_annotations_used_only_posthoc=True,
        reference_answers_used_only_posthoc=True,raw_text_or_answers_in_snapshot=False,
        selected_chunks='cached Decision API scores; approximately 10%',
        accuracy='answer_focused_lexical_v2',online_API_calls=0,
        methods='whole/selected/cartridge: same question suffix, same generation budget within each case')
    (ROOT/'data_snapshot.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Frozen records:',len(rows),';',', '.join(f'{m}: {s["completed"]}' for m,s in sources.items()))


if __name__=='__main__':main()
