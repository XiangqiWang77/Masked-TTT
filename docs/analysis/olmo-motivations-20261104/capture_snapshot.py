#!/usr/bin/env python3
"""Freeze the original 1,000-case OLMo score and support-coverage snapshot."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102/query_cartridges_20261103')


def main():
    if (ROOT/'data_snapshot.csv').exists() or (ROOT/'data_snapshot.metadata.json').exists():
        raise RuntimeError('Snapshot already exists; use a new directory for another snapshot')
    data=json.loads((PROJECT/'latest.json').read_text())
    job=next(x for x in data['jobs'] if x['name']=='olmo3-7b-babilong')
    if job['completed']!=1000 or job['all_paired_n']!=1000:
        raise ValueError('Expected the original completed 1,000-case OLMo experiment')
    if job['all_paired_accuracy']!={'vanilla':36.1,'qttt':36.0,'icl_teacher':41.2,
                                     'masked_ttt':36.1,'standalone_cartridge':44.6}:
        raise ValueError('The frozen original experiment changed')
    support=json.loads((PROJECT/'posthoc_diagnostics/selected_support.json').read_text())
    if not support['annotations_used_only_posthoc'] or support['API_calls']!=0:
        raise ValueError('Unexpected support-coverage source')
    facts={item['case']:item for item in support['records']};rows=[]
    for item in job['case_outcomes']:
        case=item['case'];bucket=int(case.split('-',2)[1]);context=item['context_coverage']
        if bucket not in (8,16,32,64):raise ValueError('Unexpected context bucket')
        if context=='complete':
            fact=facts[case]
            if item.get('support_coverage')!=fact['content_coverage']:
                raise ValueError('Coverage does not match the selected chunks')
            low=float(fact['content_fact_cover_lower']);high=float(fact['content_fact_cover_upper'])
            if low!=high or not 0<=low<=1:raise ValueError('Ambiguous support coverage')
        elif context=='compacted':low=high=None
        else:raise ValueError('Unknown original-document coverage')
        scores=item['scores']
        if any(scores[key] not in (0,1) for key in ('vanilla','icl_teacher','standalone_cartridge')):
            raise ValueError('Expected binary paired scores')
        rows.append(dict(case=case,source_question=case.split('-',2)[2],task=item['task'],
            context_tokens=bucket*1024,working_document=context,fact_cover_lower=low,
            fact_cover_upper=high,whole=int(scores['vanilla']),
            teacher=int(scores['icl_teacher']),cartridge=int(scores['standalone_cartridge'])))
    if len(rows)!=1000 or len({x['case'] for x in rows})!=1000:
        raise ValueError('Duplicate or missing cases')
    if [sum(row[k] for row in rows) for k in ('whole','teacher','cartridge')]!=[361,412,446]:
        raise ValueError('Per-case scores do not reproduce the full table')
    with (ROOT/'data_snapshot.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),model='OLMo-3-7B-Instruct',
        n=1000,source=str(PROJECT/'latest.json'),
        source_report_utc=datetime.fromtimestamp(data['updated'],timezone.utc).isoformat(),
        support_source=str(PROJECT/'posthoc_diagnostics/selected_support.json'),
        metric=data['metric'],working_document_compacted=sum(row['working_document']=='compacted' for row in rows),
        support_and_reference_labels_used_only_posthoc=True,raw_text_or_answers_in_snapshot=False,
        selection='cached Decision API scores, approximately 10%',online_API_calls=0,
        teacher_format='generic output instruction; maximum 32 teacher-generated tokens',
        cartridge_format='original question suffix and request-specific generation limit')
    (ROOT/'data_snapshot.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Frozen 1,000 OLMo records; full-document coverage:',
          sum(row['working_document']=='complete' for row in rows))


if __name__=='__main__':main()
