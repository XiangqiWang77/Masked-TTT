#!/usr/bin/env python3
"""Describe Qwen counting-answer format changes using saved paired outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.babilong_linked_selection.consolidate_per_model import consolidate


NUMBERS={'none':0,'nothing':0,'zero':0,'0':0,'one':1,'1':1,'two':2,'2':2,
         'three':3,'3':3,'four':4,'4':4,'five':5,'5':5}


def numeric_answer(text):
    value=re.sub(r'^[\s`$]+|[\s`$.,!?;:]+$','',text.casefold())
    return NUMBERS.get(value)


def format_scores(rows):
    n=len(rows)
    if not n:raise ValueError('A nonempty paired cohort is required')
    gained=sum(row['cartridge_teacher_format'] and not row['cartridge'] for row in rows)
    lost=sum(row['cartridge'] and not row['cartridge_teacher_format'] for row in rows)
    return dict(n=n,original=100*sum(row['cartridge'] for row in rows)/n,
        generic=100*sum(row['cartridge_teacher_format'] for row in rows)/n,
        teacher=100*sum(row['teacher'] for row in rows)/n,
        new_correct=gained,new_wrong=lost,delta_pp=100*(gained-lost)/n)


def summarize(rows, predictions):
    n=len(rows)
    if not n or len({row['case'] for row in rows})!=n:
        raise ValueError('Counting cases must be nonempty and unique')
    counts={key:sum(row[key] for row in rows) for key in
            ('whole','zero','teacher','cartridge','zero_teacher_format','cartridge_teacher_format')}
    gained=lost=changed_number=same_number=unparsed=0
    transitions=Counter()
    for row in rows:
        if row['cartridge']==row['cartridge_teacher_format']:continue
        gained+=int(row['cartridge_teacher_format']>row['cartridge'])
        lost+=int(row['cartridge_teacher_format']<row['cartridge'])
        raw,formatted=predictions[row['case']]
        before,after=numeric_answer(raw),numeric_answer(formatted)
        if before is None or after is None:unparsed+=1
        elif before==after:same_number+=1
        else:
            changed_number+=1
            transitions[(before,after)]+=1
    return dict(n=n,correct=counts,accuracy={key:100*value/n for key,value in counts.items()},
        generic_vs_original=dict(new_correct=gained,new_wrong=lost,delta_pp=100*(gained-lost)/n),
        score_changes_with_different_number=changed_number,
        score_changes_with_same_number=same_number,score_changes_not_parsed=unparsed,
        number_transitions={f'{a} -> {b}':count for (a,b),count in sorted(transitions.items())})


def collect_predictions(root, rows):
    result={}
    for row in rows:
        data=json.loads((root/'cases'/'qwen'/row['case']/'result.json').read_text())
        pair=data['standalone_cartridge']['paired_prompt_control']
        if (data['key']!=row['case'] or data['task']!='qa7' or
            pair['same_selected_kv'] is not True or pair['teacher_answer_in_student_prompt'] is not False or
            data['teacher_answer_in_final_prompt'] is not False or
            data['training']['objective']!='on-policy KL(teacher || student)' or
            data['training']['selected_transfer_weight']!=0):
            raise ValueError('The cached paired output does not match the original SD method')
        result[row['case']]=(pair['trained_original']['prediction'],
                             pair['trained_teacher_suffix']['prediction'])
    return result


def audit(development_manifest,holdout_manifest,development,holdout,development_root,holdout_root):
    combined=consolidate(development_manifest,holdout_manifest,development,holdout)
    qwen=combined['models']['qwen']
    if qwen['state']!='complete':raise ValueError('Qwen must complete the fixed 1,000 cases')
    dev_rows=[row for row in development['models']['qwen']['rows'] if row['task']=='qa7']
    held_rows=[row for row in holdout['models']['qwen']['rows'] if row['task']=='qa7']
    if len(dev_rows)!=20 or len(held_rows)!=80:
        raise ValueError('Unexpected development or holdout counting cohort')
    predictions=collect_predictions(development_root,dev_rows)
    predictions.update(collect_predictions(holdout_root,held_rows))
    all_rows=development['models']['qwen']['rows']+holdout['models']['qwen']['rows']
    counting_rows=dev_rows+held_rows
    return dict(metric=combined['metric'],selection_policy=combined['selection_policy'],
        development_used_for_selection=True,source_disjoint_holdout=True,
        teacher_has_extra_generic_format_instruction=True,
        generic_answer_is_pre_release_diagnostic=True,
        online_api_calls=0,model_calls=0,reference_answers_read_by_this_script=False,
        evaluation_answers_used_only_posthoc=True,
        total_teacher_minus_original_correct=sum(row['teacher']-row['cartridge'] for row in all_rows),
        qa7_teacher_minus_original_correct=sum(row['teacher']-row['cartridge'] for row in counting_rows),
        all_tasks_format=format_scores(all_rows),
        qa4_format=format_scores([row for row in all_rows if row['task']=='qa4']),
        development=summarize(dev_rows,predictions),holdout=summarize(held_rows,predictions),
        combined=summarize(dev_rows+held_rows,predictions))


def render(data):
    def row(name, part):
        a=part['accuracy'];p=part['generic_vs_original']
        return (f"| {name} | {part['n']} | {a['zero']:.2f} | {a['teacher']:.2f} | "
                f"{a['cartridge']:.2f} | {a['cartridge_teacher_format']:.2f} | "
                f"{p['new_correct']} / {p['new_wrong']} |")
    lines=['# Qwen 计数题：回答格式与训练信号','',
        '| 范围 | 题数 | 原格式零更新 | Teacher | 原格式卡带 | 同训练状态＋通用格式 | 新对 / 新错 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for part in ('development','holdout','combined'):lines.append(row(part,data[part]))
    c=data['combined']
    overall=data['all_tasks_format'];qa4=data['qa4_format']
    lines+=['',f"在 1,000 题中，Teacher 相对原格式卡带净多答对 {data['total_teacher_minus_original_correct']} 题；"
        f"其中计数题贡献 {data['qa7_teacher_minus_original_correct']} 题。",'',
        f"计数题中切换通用格式造成 {c['generic_vs_original']['new_correct']} 个新对、"
        f"{c['generic_vs_original']['new_wrong']} 个新错；"
        f"其中 {c['score_changes_with_different_number']} 个评分变化对应模型直接输出不同的数字，"
        f"{c['score_changes_not_parsed']} 个不能按单一数字解析。",'',
        f"全部 1,000 题使用通用格式时为 {overall['generic']:.2f}%，"
        f"原格式为 {overall['original']:.2f}%，Teacher 为 {overall['teacher']:.2f}%。"
        f"其中 QA4 从 {qa4['original']:.2f}% 变为 {qa4['generic']:.2f}%，"
        '因此格式影响随问题类型变化。', '',
        '同格式结果来自相同训练状态、相同选中 KV 的释放前配对诊断；Teacher 答案没有加入 Student 提示。'
        'Qwen 真实 B200 的合成预检已检查释放前后该格式一致，逐题释放后的结果仍需独立运行。', '',
        '开发集用于配置选择，留出来源问题不重叠；两道留出来源问题此前做过定性查看，旧配置总体也已查看。'
        '这些是事后诊断，不把按题型选择格式的假设当作新的盲测收益。', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    dev=args.workspace/'query_cartridges_linked_selection_20261104'
    held=args.workspace/'query_cartridges_linked_holdout_20261104'
    paths=(dev/'manifest.json',held/'manifest.json',held/'development_snapshot.json',held/'latest.json')
    data=audit(*(json.loads(path.read_text()) for path in paths),dev,held)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    (args.output_dir/'README.md').write_text(render(data))
    print(json.dumps(data['combined'],ensure_ascii=False))


if __name__=='__main__':main()
