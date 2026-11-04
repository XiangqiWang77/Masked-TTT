#!/usr/bin/env python3
"""Freeze matched 1,000-case BABILONG tables from existing evaluated outputs."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
WORK = Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/cross_backbone_selective_sd_20261102')
REFS = Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/babilong_dense_evidence_20261029/v3/evaluation_references.json')
OLMO = WORK/'query_cartridges_zero_control_20261103'
QWEN = WORK/'query_cartridges_qwen_olmo1000_20261104'
GPT = WORK/'query_cartridges_gpt_olmo1000_20261103'
SNAPSHOT = REPO/'docs/analysis/olmo-motivations-20261104/data_snapshot.csv'
LENGTHS = (8192, 16384, 32768, 65536)
LABELS = dict(olmo='OLMo-3-7B', qwen='Qwen3-4B', gpt='GPT-OSS-20B')


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError('Empty evaluation group')
    names = ('whole', 'teacher', 'zero', 'cartridge')
    correct = {name: sum(int(row[name]) for row in rows) for name in names}
    result = dict(n=n, correct=correct,
                  accuracy={name:100*correct[name]/n for name in names})
    result['vs_teacher'] = dict(new_correct=sum(not r['teacher'] and r['cartridge'] for r in rows),
                                new_wrong=sum(r['teacher'] and not r['cartridge'] for r in rows),
                                delta_pp=100*(correct['cartridge']-correct['teacher'])/n)
    result['vs_zero'] = dict(new_correct=sum(not r['zero'] and r['cartridge'] for r in rows),
                             new_wrong=sum(r['zero'] and not r['cartridge'] for r in rows),
                             delta_pp=100*(correct['cartridge']-correct['zero'])/n)
    return result


def load():
    sys.path.insert(0, str(WORK/'monitor'))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, score_prediction
    if BABILONG_SCORER_VERSION != 'answer_focused_lexical_v2':
        raise ValueError('Unexpected scorer')
    refs = json.loads(REFS.read_text())
    with SNAPSHOT.open(newline='') as stream:
        snapshot = list(csv.DictReader(stream))
    if len(snapshot) != 1000 or len({r['case'] for r in snapshot}) != 1000:
        raise ValueError('The OLMo snapshot is incomplete')
    keys = {r['case'] for r in snapshot}
    by_model = {'olmo': []}
    for row in snapshot:
        case = row['case']
        result = json.loads((OLMO/'cases'/case/'paired.json').read_text())
        ref, contract = refs[case], result['contract']
        if (result['key'] != case or result['model'] != 'allenai/Olmo-3-7B-Instruct' or
            result['reference_answers_used'] or contract['reference_answers_used'] or
            not contract['same_selected_kv'] or not contract['same_question_suffix'] or
            not contract['same_backbone_and_device'] or contract['control_update_steps'] != 0):
            raise ValueError('OLMo pairing contract changed')
        def score(name):
            return int(score_prediction(result[name], ref['answers'], 'babilong', ref['task'])[0])
        by_model['olmo'].append(dict(case=case, task=row['task'], length=int(row['context_tokens']),
            whole=int(row['whole']), teacher=score('teacher'), zero=score('zero_update'),
            cartridge=score('trained')))
    for name, path in (('qwen', QWEN), ('gpt', GPT)):
        report = json.loads((path/'latest.json').read_text())
        if (report['metric'] != 'answer_focused_lexical_v2' or report['completed'] != 1000 or
            report['errors'] or report['reference_answers_used_only_posthoc'] is not True or
            {row['case'] for row in report['rows']} != keys):
            raise ValueError('Model report is not the same completed 1,000-case cohort')
        by_model[name] = [dict(case=r['case'], task=r['task'], length=int(r['length']),
            whole=int(r['whole']), teacher=int(r['teacher']), zero=int(r['selective']),
            cartridge=int(r['cartridge'])) for r in report['rows']]
    expected = {'olmo': (361, 410, 443, 445), 'qwen': (544, 566, 560, 567),
                'gpt': (487, 571, 551, 554)}
    for name, rows in by_model.items():
        if tuple(sum(r[k] for r in rows) for k in ('whole','teacher','zero','cartridge')) != expected[name]:
            raise ValueError('An audited aggregate changed')
    return by_model


def table(rows):
    header = '| ' + ' | '.join(rows[0]) + ' |'
    separator = '| ' + ' | '.join('---' if i == 0 else '---:' for i in range(len(rows[0]))) + ' |'
    return '\n'.join([header, separator] + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows[1:]])


def main():
    models = load()
    summary = dict(metric='answer_focused_lexical_v2', n_cases=1000,
        source_questions=250, context_lengths=LENGTHS, evaluation_answers_used_only_posthoc=True,
        model_or_api_calls=0,
        selection_rule='minimum ICL among the original and linked 1,000-case configurations and OLMo matched rerun; report the paired QC from the same run',
        olmo_full_baseline_source='original run on the same case IDs, not regenerated in the zero-update rerun',
        teacher_and_cartridge_prompt_formats_differ=True, models={})
    total_table = [['模型','全文 KV','ICL Teacher','选中 KV 零更新','Query Cartridge','相对 Teacher','相对零更新']]
    length_table = [['模型 / 长度','题数','全文 KV','ICL Teacher','零更新','Query Cartridge','相对 Teacher']]
    task_table = [['题型（每组100题）','OLMo：ICL → QC（Δ）','Qwen：ICL → QC（Δ）','GPT-OSS：ICL → QC（Δ）']]
    for name, rows in models.items():
        overall = summarize(rows)
        by_length = {str(n):summarize([r for r in rows if r['length'] == n]) for n in LENGTHS}
        by_task = {f'qa{i}':summarize([r for r in rows if r['task'] == f'qa{i}']) for i in range(1,11)}
        if any(group['n'] != 250 for group in by_length.values()) or any(group['n'] != 100 for group in by_task.values()):
            raise ValueError('Unexpected question grouping')
        summary['models'][name] = dict(overall=overall, by_length=by_length, by_task=by_task)
        a=overall['accuracy']
        total_table.append([LABELS[name],f"{a['whole']:.2f}%"+('*' if name=='olmo' else ''),
            f"{a['teacher']:.2f}%",f"{a['zero']:.2f}%",f"{a['cartridge']:.2f}%",
            f"{overall['vs_teacher']['delta_pp']:+.2f}",f"{overall['vs_zero']['delta_pp']:+.2f}"])
        for length, group in by_length.items():
            a=group['accuracy']
            length_table.append([f"{LABELS[name]} · {int(length)//1024}K",str(group['n']),
                f"{a['whole']:.1f}"+('*' if name=='olmo' else ''),f"{a['teacher']:.1f}",
                f"{a['zero']:.1f}",f"{a['cartridge']:.1f}",f"{group['vs_teacher']['delta_pp']:+.1f}"])
    for i in range(1,11):
        line=[f'QA{i}']
        for name in ('olmo','qwen','gpt'):
            g=summary['models'][name]['by_task'][f'qa{i}']; a=g['accuracy']
            line.append(f"{a['teacher']:.0f} → {a['cartridge']:.0f} ({g['vs_teacher']['delta_pp']:+.0f})")
        task_table.append(line)
    prose = [
        '# Query Cartridges：BABILONG 完整结果', '',
        '每模型同一批 1,000 次请求：250 个来源问题各对应 8K、16K、32K、64K 四种上下文。',
        '均使用 `answer_focused_lexical_v2` 宽松词面评分；百分数为正确率，差值为百分点。',
        '没有新模型或 API 调用。', '',
        '## 一、整体：同一运行内的对照', '',
        '按用户要求，在本报告核对的完整 1,000 题设置中选取各模型较低的 ICL Teacher，',
        '并且始终报告该次运行对应的 Cartridge；没有从不同 Teacher 混搭每题结果。',
        '范围包括原始选块、关联选块及 OLMo 的配对复跑，并不声称扫描过所有历史实验。', '',
        table(total_table), '',
        '*OLMo 全文 KV 取同 1,000 个题目 ID 的原始运行；该配对复跑未另存全文回答。',
        'OLMo 的 64K 组有 162 题由于窗口限制先压缩工作文档。',
        'Teacher 使用额外通用输出格式要求，而 QC 使用原问题格式；这列差值包含格式和其他设置差异。', '',
        '## 二、按上下文长度', '',
        '每个长度组 250 题；表内准确率单位为 %，差值为百分点。', '', table(length_table), '',
        '## 三、按题型', '',
        '每个题型 100 题；括号是 QC 相对同轮 Teacher 的百分点变化。', '', table(task_table), '',
        '## 四、其他完整选块设置', '',
        '| 模型 / 选块 | 题数 | 全文 KV | ICL Teacher | 零更新 | Query Cartridge | 相对 Teacher |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        '| OLMo / 原始运行 | 1,000 | 36.10% | 41.20% | — | 44.60% | +3.40 |',
        '| Qwen / 关联选块 | 1,000 | 54.70% | 61.30% | 59.40% | 60.20% | −1.10 |',
        '| GPT-OSS / 关联选块 | 1,000 | 48.80% | 62.60% | 60.40% | 60.30% | −2.30 |', '',
        '关联选块结果由 200 道开发题和 800 道来源不重叠的留出题组成；',
        '选块和 Teacher 与第一表不同，不能将该配置的 QC 与另一配置的最低 Teacher 拼接成配对涨点。', '',
        '## 五、同缓存、同格式的更新对照', '',
        '| 模型 / 格式 | 零更新 | 训练后 | 差值 |',
        '| --- | ---: | ---: | ---: |',
        '| OLMo / 原问题格式 | 44.30% | 44.50% | +0.20 |',
        '| Qwen / Teacher 格式 | 56.60% | 56.80% | +0.20 |',
        '| GPT-OSS / Teacher 格式 | 57.10% | 57.50% | +0.40 |', '',
        '同格式对照用于单独观察更新，和第一表的正式 QC 输出格式不总是相同。',
        '表中的历史配对运行未单独测量物理释放后的总显存峰值。', '',
        '[逐项数字](summary.json) · [生成脚本](build_report.py) · '
        '[关联选块来源](../three-model-babilong-20261104/README.md) · '
        '[题型配对来源](../olmo-task-reproduction-20261104/README.md)', ''
    ]
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'README.md').write_text('\n'.join(prose))
    print(table(total_table));print();print(table(length_table));print();print(table(task_table))


if __name__ == '__main__':
    main()
