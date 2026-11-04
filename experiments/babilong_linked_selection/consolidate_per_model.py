#!/usr/bin/env python3
"""Consolidate each completed 200+800 split without waiting for other models."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.babilong_linked_selection.paired_attribution import ARMS, summarize
from experiments.babilong_linked_selection.snapshot_holdout import freeze, source_id


def validate_rows(rows, expected, *, complete):
    keys = [row['case'] for row in rows]
    if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
        raise ValueError('Duplicate or out-of-scope cases')
    if complete and set(keys) != expected:
        raise ValueError('Completed model does not cover its exact split')
    for row in rows:
        key = row['case']
        parts = key.split('-', 2)
        if (row['task'] != source_id(key).split('-', 1)[0] or
            row['length'] != int(parts[1])*1024 or row['baseline_source'] != 'full_1000' or
            type(row['same_full_answer']) is not bool or
            any(type(row.get(arm)) is not int or row[arm] not in (0, 1) for arm in ARMS)):
            raise ValueError('A paired case has inconsistent task, length, baseline or score')


def consolidate(development_manifest, holdout_manifest, development, holdout):
    if (development_manifest['mode'] != 'development' or development_manifest['case_count'] != 200 or
        development['development_only'] is not True or development['expected_per_model'] != 200 or
        development['metric'] != 'answer_focused_lexical_v2' or development['errors'] or
        development['invalid'] or holdout['errors'] or holdout['invalid'] or
        development['selected_fraction'] != .1 or development['baseline_fraction'] != .1 or
        development['selection_policy'] != 'decision_entity_link_v1' or
        development['cached_scores'] is not True or development['online_api_calls'] != 0 or
        development['evaluation_answers_used_only_posthoc'] is not True or
        set(development_manifest['holdout_case_ids']) != set(holdout_manifest['case_ids'])):
        raise ValueError('The fixed development or holdout protocol changed')
    for manifest in (development_manifest, holdout_manifest):
        if (manifest['models'] != ['qwen', 'gpt'] or manifest['keep_fraction'] != .1 or
            manifest['selection_policy'] != 'decision_entity_link_v1' or
            manifest['entity_reserve_fraction'] != .25 or manifest['basis_source'] != 'document' or
            manifest['selected_transfer_weight'] != 0 or
            manifest['reference_answers_used_in_inference'] is not False or manifest['online_api_calls'] != 0):
            raise ValueError('The model, selection or SD settings changed')
    frozen = freeze(holdout_manifest, development_manifest, holdout)
    dev_keys = set(development_manifest['case_ids'])
    held_keys = set(holdout_manifest['case_ids'])
    models = {}
    for model in ('qwen', 'gpt'):
        dev_rows = development['models'][model]['rows']
        held_rows = holdout['models'][model]['rows']
        validate_rows(dev_rows, dev_keys, complete=True)
        validate_rows(held_rows, held_keys, complete=False)
        if len(held_rows) != 800:
            models[model] = dict(state='in_progress', development_count=len(dev_rows),
                holdout_count=len(held_rows), holdout_partial=frozen['models'][model]['overall'])
            continue
        validate_rows(held_rows, held_keys, complete=True)
        rows = dev_rows + held_rows
        if len(rows) != 1000 or len({row['case'] for row in rows}) != 1000:
            raise ValueError('Combined model must cover 1,000 distinct cases')
        groups = defaultdict(list)
        for row in rows:
            groups['task:' + row['task']].append(row)
            groups['length:' + str(row['length'])].append(row)
        models[model] = dict(state='complete', development=summarize(dev_rows),
            holdout=summarize(held_rows), combined=summarize(rows),
            groups={key:summarize(items) for key,items in sorted(groups.items())},
            same_full_answer=sum(row['same_full_answer'] for row in rows),
            holdout_excluding_previous_examples=frozen['models'][model]['excluding_previously_viewed_sources'])
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), metric=holdout['metric'],
        selected_fraction=.1, selection_policy='decision_entity_link_v1',
        development_used_for_selection=True, source_disjoint_holdout=True,
        teacher_has_extra_generic_format_instruction=True,
        same_format_diagnostics_before_source_release=True,
        online_api_calls=0, model_calls=0, evaluation_answers_used_only_posthoc=True,
        models=models)


def render(data):
    def fmt(value):
        return '—' if value is None else f'{value:.2f}'
    lines=['# BABILONG：逐模型 1,000 题配对结果', '',
        '生成时间：'+data['created_utc']+'。进行中的模型只显示截至该时间的留出集进度。', '',
        '| 模型 / 范围 | 题数 | 全文 | 原分数选块卡带 | 关联选块零更新 | 关联选块 Teacher | 关联选块卡带 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def row(label, summary):
        a=summary['accuracy']
        return f"| {label} | {summary['n']} | {fmt(a['whole'])} | {fmt(a['baseline10'])} | {fmt(a['zero'])} | {fmt(a['teacher'])} | {fmt(a['cartridge'])} |"
    for name,item in data['models'].items():
        if item['state']=='complete':
            for part in ('development','holdout','combined'):
                lines.append(row(name+' '+part,item[part]))
            for i in range(1,11):
                lines.append(row(name+' QA'+str(i),item['groups']['task:qa'+str(i)]))
            for size in (8192,16384,32768,65536):
                lines.append(row(name+' '+str(size//1024)+'K',item['groups']['length:'+str(size)]))
        else:
            lines.append(row(name+' 留出集进行中',item['holdout_partial']))
    lines += ['', '指标为同一套宽松词面评分。开发集用于选择配置，800 题留出集与开发集按来源问题分开。',
        '留出集中两个来源问题此前做过定性查看；另一个旧选块配置的整批汇总也曾被查看，因此留出集并非完全盲测。',
        'Teacher 加了通用回答格式要求；原格式零更新与原格式训练后卡带使用相同提示。',
        '原分数选块卡带与关联选块卡带使用相同块数预算，但选块、Teacher 和训练轨迹可能不同。',
        '同格式卡带属于释放前配对诊断；真实模型物理释放核验单独报告。', '']
    for name,item in data['models'].items():
        if item['state'] != 'complete':continue
        overall=item['combined'];a=overall['accuracy'];p=overall['pairs']
        lines.append(f"{name} 合计：相对 Teacher {a['cartridge']-a['teacher']:+.2f} 点；"
            f"相对同格式零更新 {p['sd_original_format']['delta_pp']:+.2f} 点"
            f"（新对 {p['sd_original_format']['new_correct']} / 新错 {p['sd_original_format']['new_wrong']}）；"
            f"相对原选块卡带 {p['selection_and_method']['delta_pp']:+.2f} 点。")
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    dev=args.workspace/'query_cartridges_linked_selection_20261104'
    held=args.workspace/'query_cartridges_linked_holdout_20261104'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'report.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=consolidate(*(json.loads(path.read_text()) for path in
            (dev/'manifest.json',held/'manifest.json',held/'development_snapshot.json',held/'latest.json')))
        for name,content in (('summary.json',json.dumps(data,indent=2,ensure_ascii=False)+'\n'),
                             ('report.md',render(data))):
            path=args.output_dir/name
            tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(content);tmp.replace(path)
    for name,item in data['models'].items():
        print(name,item['state'],item.get('holdout_count',800))


if __name__=='__main__':main()
