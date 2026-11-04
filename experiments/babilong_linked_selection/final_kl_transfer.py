#!/usr/bin/env python3
"""Audit the last fixed-prefix KL update against paired standalone answers."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.babilong_linked_selection.consolidate_per_model import consolidate


EPSILON=1e-6


def direction(before,after):
    delta=before-after
    return 'decreased' if delta>EPSILON else 'increased' if delta< -EPSILON else 'within_tolerance'


def summarize(rows):
    n=len(rows)
    if not n:return dict(n=0)
    directions=Counter(direction(row['before'],row['after']) for row in rows)
    gain=sum(not row['zero'] and row['cartridge'] for row in rows)
    loss=sum(row['zero'] and not row['cartridge'] for row in rows)
    return dict(n=n,kl_decreased=directions['decreased'],kl_increased=directions['increased'],
        kl_within_tolerance=directions['within_tolerance'],
        median_kl_before=statistics.median(row['before'] for row in rows),
        median_kl_after=statistics.median(row['after'] for row in rows),
        median_kl_reduction=statistics.median(row['before']-row['after'] for row in rows),
        zero_accuracy=100*sum(row['zero'] for row in rows)/n,
        cartridge_accuracy=100*sum(row['cartridge'] for row in rows)/n,
        teacher_accuracy=100*sum(row['teacher'] for row in rows)/n,
        new_correct=gain,new_wrong=loss,delta_pp=100*(gain-loss)/n)


def audit(development_manifest,holdout_manifest,development,holdout,development_root,holdout_root):
    validated=consolidate(development_manifest,holdout_manifest,development,holdout)
    if validated['models']['qwen']['state']!='complete':
        raise ValueError('Qwen must complete all 1,000 cases')
    protocols=[json.loads(Path(manifest['model_protocols']['qwen']).read_text())
               for manifest in (development_manifest,holdout_manifest)]
    if any(protocol[key]!=protocols[0][key] for protocol in protocols for key in ('index','source_cases')):
        raise ValueError('The paired source records differ')
    entries={row['key']:row for row in json.loads(Path(protocols[0]['index']).read_text())}
    records=[]
    with Path(protocols[0]['source_cases']).open('rb') as source:
        for split,report,root in (('development',development,development_root),('holdout',holdout,holdout_root)):
            for row in report['models']['qwen']['rows']:
                key=row['case'];entry=entries[key]
                source.seek(entry['offset']);original=json.loads(source.read(entry['bytes']))
                if (original.get('key')!=key or 'Choices:' in original['query'] or
                    set(original)&{'answer','answers','references','gold','answer_spans'}):
                    raise ValueError('Expected the original answer-free non-choice request')
                result=json.loads((root/'cases'/'qwen'/key/'result.json').read_text())
                training=result['training'];steps=training['steps'];verified=result['self_study']['verified']
                if (result['key']!=key or result['task']!=row['task'] or not steps or
                    training['objective']!='on-policy KL(teacher || student)' or
                    training['selected_transfer_weight']!=0 or
                    training['accepted_steps']!=len(steps) or len(steps)!=2+verified or
                    not training['teacher_conditioned_on_student_prefix'] or
                    not training['student_rollouts_refreshed_each_step'] or
                    result['teacher_answer_in_final_prompt'] or not result['backbone_unchanged'] or
                    not result['original_document_KV_unchanged']):
                    raise ValueError('The logged on-policy update contract differs')
                last=steps[-1];before=last['teacher_KL_before'];after=training['final_kl']
                if (last['question_path']!='main' or last['step']!=len(steps) or
                    last['target_positions']!=list(range(len(last['student_rollout_tokens']))) or
                    not all(type(value) in (float,int) and math.isfinite(value) for value in (before,after))):
                    raise ValueError('The last update does not use the full fixed main-question prefix')
                records.append(dict(split=split,task=row['task'],length=row['length'],
                    before=before,after=after,zero=row['zero'],teacher=row['teacher'],cartridge=row['cartridge']))
    if len(records)!=1000:raise ValueError('Expected the complete paired Qwen cohort')
    grouped=defaultdict(list)
    for record in records:
        grouped['split:'+record['split']].append(record)
        grouped['task:'+record['task']].append(record)
        grouped['last_kl:'+direction(record['before'],record['after'])].append(record)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),model='Qwen3-4B-Instruct-2507',
        metric=validated['metric'],scope='last accepted main-question update on the same logged prefix',
        epsilon=EPSILON,online_api_calls=0,model_calls=0,reference_answers_read_by_this_script=False,
        evaluation_answers_used_only_posthoc=True,development_used_for_selection=True,
        source_disjoint_holdout=True,overall=summarize(records),
        groups={key:summarize(rows) for key,rows in sorted(grouped.items())})


def render(data):
    def line(label,x):
        if not x['n']:return f"| {label} | 0 | — | — | — | — | — |"
        return (f"| {label} | {x['n']} | {x['kl_decreased']} | {x['kl_within_tolerance']} | "
                f"{x['kl_increased']} | {x['median_kl_reduction']:.6f} | "
                f"{x['new_correct']} / {x['new_wrong']} |")
    lines=['# Qwen：最后一步蒸馏 KL 与独立卡带答案','',
        '| 范围 | 题数 | KL 下降 | 变化 ≤ 1e−6 | KL 上升 | KL 降幅中位数 | 卡带相对零更新：新对 / 新错 |',
        '|---|---:|---:|---:|---:|---:|---:|',line('全部',data['overall'])]
    for key in ('split:development','split:holdout','last_kl:decreased',
                'last_kl:within_tolerance','last_kl:increased'):
        if key in data['groups']:lines.append(line(key,data['groups'][key]))
    lines+=['','每题比较最后一次已接受更新前后的 KL，使用同一个主问题、同一学生前缀和同一 Teacher 分布；'
        '检查排除了选择题式提前停止。答案变化比较同一选中缓存、同一原问题格式的零更新和独立卡带。',
        '', 'KL 下降说明该局部训练目标改善，不等同于最终答案一定改善；不同题的绝对 KL 不可直接视作正确性分数。'
        '数据为事后分析：开发集用于选择配置，留出来源问题不重叠，但此前曾查看两个来源问题和旧配置汇总。',
        '', '脚本不调用模型或 API，不读取参考答案；已有二值答案评分仅用于事后分组。', '',
        '[聚合数据](summary.json) · [复核脚本](../../../experiments/babilong_linked_selection/final_kl_transfer.py)', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();dev=args.workspace/'query_cartridges_linked_selection_20261104'
    held=args.workspace/'query_cartridges_linked_holdout_20261104'
    paths=(dev/'manifest.json',held/'manifest.json',held/'development_snapshot.json',held/'latest.json')
    data=audit(*(json.loads(path.read_text()) for path in paths),dev,held)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    (args.output_dir/'README.md').write_text(render(data))
    print(json.dumps(data['overall'],ensure_ascii=False))


if __name__=='__main__':main()
