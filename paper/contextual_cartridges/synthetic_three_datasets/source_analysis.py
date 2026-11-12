"""Plot two observable synthetic-record diagnostics on a matched request cohort.

No model, SD result, benchmark answer, or semantic judge enters either metric.
These are query-term focus and disjoint-source co-presentation, not correctness
or a demonstration that a question requires multi-hop reasoning.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_synthetic_comparison import query_terms, normalized
from experiments.run_figure4_contextual import load_cases

DATASETS = ('longhealth', 'mtob_ke', 'qasper')
NAMES = {'longhealth': 'LongHealth', 'mtob_ke': 'MTOB (KE)', 'qasper': 'QASPER'}
ARMS = ('vanilla', 'specific')
METRICS = ('query_term_focus', 'disjoint_source_context')
BLUE, ORANGE, INK = '#3180A8', '#E36A1C', '#0B2A3A'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')


def merged(spans):
    out = []
    for start, end in sorted(spans):
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end):
            raise ValueError('Invalid source span')
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def source_spans(example, arm, document):
    if arm == 'specific':
        chunks = example['evidence']['source_chunks']
        for chunk in chunks:
            start, end = chunk['span']
            if document[start:end] != chunk['text']:
                raise ValueError('Stored source coordinates do not match the document')
        spans = merged([c['span'] for c in chunks])
    else:
        prompt = example['source_system_prompt']
        prefix = '\nYou are in a conversation about the following user information.\n\n<info>\n'
        suffix = '\n</info>'
        if not prompt.startswith(prefix) or not prompt.endswith(suffix):
            raise ValueError('Unexpected official self-study context wrapper')
        content = prompt[len(prefix):-len(suffix)]
        start = document.find(content)
        if start < 0 or not content:
            raise ValueError('Vanilla context is not an exact source substring')
        if document.find(content, start + 1) >= 0:
            raise ValueError('Ambiguous source location; do not choose silently')
        spans = [[start, start + len(content)]]
    # Adjacent source chunks do not constitute separated source regions.
    return spans


def collect(campaign):
    config = read(campaign / 'config.json')
    plan = read(campaign / 'plan.json')
    unique = {}
    for row in plan:
        key = (row['dataset'], row['case'])
        if key in unique:
            if any(row[k] != unique[key][k] for k in
                   ('data_files', 'data_sha256', 'synthetic_questions', 'document_sha256')):
                raise ValueError('Cache-size rows reference different synthetic data')
        else:
            unique[key] = row
    records, case_rows, inputs = [], [], {}
    for dataset in DATASETS:
        sources = {c['id']: c for c in load_cases(dataset, config['dataset_root'])}
        for row in [r for r in unique.values() if r['dataset'] == dataset]:
            case = sources[row['case']]
            document = case['document']
            if hashlib.sha256(document.encode()).hexdigest() != row['document_sha256']:
                raise ValueError('Document identity mismatch')
            data = {}
            for arm, key in [('vanilla', 'agnostic_json'), ('specific', 'specific_json')]:
                path = Path(row['data_files'][key])
                fingerprint = sha(path)
                if fingerprint != row['data_sha256'][key]:
                    raise ValueError('Synthetic source changed since the training plan was frozen')
                inputs[str(path)] = fingerprint
                data[arm] = read(path)
            s, v = data['specific'], data['vanilla']
            if s['query'] != case['query'] or s['case'] != row['case']:
                raise ValueError('Query identity mismatch')
            if v['query_conditioned'] or v['document_sha256'] != row['document_sha256']:
                raise ValueError('Vanilla data identity mismatch')
            n = row['synthetic_questions']
            if len(s['examples']) != n or len(v['questions']) < n or n < 1:
                raise ValueError('The matched training question counts are inconsistent')
            query = s['query']
            q_terms = set(query_terms(query))
            if not q_terms:
                raise ValueError('No query terms; record is ineligible, not a zero')
            fallback = bool(s['original_query_teacher_fallback'])
            for arm, examples in [('vanilla', v['questions'][:n]), ('specific', s['examples'])]:
                local = []
                for i, example in enumerate(examples):
                    question = example['question']
                    # The stored fallback includes task instructions / options.
                    # Measure the actual question, not these known wrappers.
                    metric_question = query if (arm == 'specific' and fallback) else question
                    hits = sorted(q_terms & set(query_terms(metric_question)))
                    spans = source_spans(example, arm, document)
                    gaps = [b[0] - a[1] for a, b in zip(spans, spans[1:])]
                    raw_id = f"{row['data_files']['agnostic_json' if arm == 'vanilla' else 'specific_json']}#{i}"
                    item = dict(dataset=dataset, case=row['case'], document_key=row['document_key'],
                        arm=arm, question_index=i, question=question, metric_question=metric_question,
                        query=query, query_terms=sorted(q_terms), matched_query_terms=hits,
                        query_term_focus=bool(hits), query_term_recall=len(hits) / len(q_terms),
                        disjoint_source_context=len(spans) >= 2, source_regions=spans,
                        gap_characters=gaps, source_characters=sum(b-a for a, b in spans),
                        raw_record_id=raw_id, paired_request_used_query_fallback=fallback,
                        is_query_fallback=arm == 'specific' and fallback,
                        exact_query_copy=normalized(metric_question) == normalized(query))
                    records.append(item)
                    local.append(item)
                case_rows.append(dict(dataset=dataset, case=row['case'], arm=arm,
                    document_key=row['document_key'], questions=n, paired_request_used_query_fallback=fallback,
                    **{k: statistics.mean(x[k] for x in local) for k in
                       (*METRICS, 'query_term_recall', 'exact_query_copy')}))
    return records, case_rows, inputs


def summarize(case_rows, records):
    out = {}
    for dataset in DATASETS:
        subset = [r for r in case_rows if r['dataset'] == dataset]
        vr = [r for r in subset if r['arm'] == 'vanilla']
        out[dataset] = dict(requests=len(vr),
            documents=len({r['document_key'] for r in vr}),
            matched_exercise_exposures_per_arm=sum(r['questions'] for r in vr),
            query_fallback_requests=sum(r['paired_request_used_query_fallback'] for r in vr),
            arms={})
        for arm in ARMS:
            rows = [r for r in subset if r['arm'] == arm]
            generated = [r for r in rows if not r['paired_request_used_query_fallback']]
            actual = [r for r in records if r['dataset'] == dataset and r['arm'] == arm]
            out[dataset]['arms'][arm] = dict(
                **{metric: 100 * statistics.mean(r[metric] for r in rows)
                   for metric in (*METRICS, 'query_term_recall', 'exact_query_copy')},
                unique_source_records=len({r['raw_record_id'] for r in actual}),
                excluding_query_fallback=dict(requests=len(generated), **{
                    metric: 100 * statistics.mean(r[metric] for r in generated)
                    if generated else None for metric in METRICS}))
    return out


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image

    style = {'font.family': 'DejaVu Sans', 'font.size': 9,
             'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK,
             'ytick.color': INK, 'axes.edgecolor': INK, 'axes.linewidth': .8,
             'hatch.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'svg.fonttype': 'none', 'figure.facecolor': 'white',
             'axes.facecolor': 'white', 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.6))
        fig.subplots_adjust(left=.14, right=.97, bottom=.10, top=.85, hspace=.56)
        titles = ('(a) Dilution: query-term focus ↑',
                  '(b) Fragmentation: non-adjacent sources ↑')
        for ax, metric, title in zip(axes, METRICS, titles):
            for arm, offset, color in zip(ARMS, (-.16, .16), (BLUE, ORANGE)):
                values = [summary[d]['arms'][arm][metric] for d in DATASETS]
                xs = [i + offset for i in range(3)]
                ax.bar(xs, values, width=.28, color=color,
                       edgecolor=INK, linewidth=.4, zorder=3)
                for x, value in zip(xs, values):
                    ax.text(x, value + 2., f'{value:.1f}' if value else '0',
                            ha='center', va='bottom', fontsize=8.7, color=INK)
            ax.set_xlim(-.55, 2.55)
            ax.set_ylim(0, 112)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xticks(range(3), [NAMES[d] for d in DATASETS])
            ax.set_ylabel('Exercises (%)', labelpad=8)
            ax.set_title(title, loc='left', fontsize=10, pad=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='x', length=0, pad=7)
            ax.tick_params(axis='y', length=3)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#D8E2E8', linewidth=.55)
        methods = [Patch(facecolor=BLUE, edgecolor=INK, linewidth=.4, label='Vanilla Cartridges'),
                   Patch(facecolor=ORANGE, edgecolor=INK, linewidth=.4, label='Contextual Cartridges')]
        fig.legend(handles=methods, loc='upper center', bbox_to_anchor=(.55, .975),
                   ncol=2, frameon=False, fontsize=9, handlelength=1.4, columnspacing=1.7)
        stem = output / 'synthetic_dilution_fragmentation'
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(stem.with_suffix('.' + ext), dpi=400, facecolor='white')
        plt.close(fig)
    with Image.open(stem.with_suffix('.png')) as im:
        export = dict(pixels=list(im.size), dpi=list(im.info.get('dpi', ())),
                      mode=im.mode, square=im.width == im.height)
    return dict(matplotlib_version=matplotlib.__version__,
                width_inches=5.6, height_inches=5.6, png=export,
                colors={'vanilla': BLUE, 'specific': ORANGE, 'text': INK},
                intervals='none', y_axis_zero_baseline=True,
                publication_format='provisional general manuscript figure; no venue compliance claim')


def report(summary, output, manifest):
    lines = [
        '# Synthetic dataset records: dilution and fragmentation', '',
        '![Matched synthetic-record comparison on three datasets](synthetic_dilution_fragmentation.png)', '',
        '只比较当前训练实际复用的 synthetic records，不包含 SD、JSD 或下游回答得分。', '',
        '**两项均为正方向指标，但都不是语义正确率。**', '',
        '- **Query-term focus ↑（Dilution 代理）**：练习问句至少命中一个真实问题的非停用词。'
        '复用之前 BABILong 记录分析的分词和停用词规则；大小写归一化，整词匹配。'
        '只检查问句，不把答案或材料中的关键词算作命中。该指标会受到共同人名、论文标题、同义词和跨语言表达影响。',
        '- **Non-adjacent sources ↑（Fragmentation 结构指标）**：同一道练习绑定至少两个在原文中不相邻的片段。'
        '逐字核对来源坐标，并合并重叠及首尾相接的片段。绑定材料一起出现不等于题目必须联合推理，也不等于材料覆盖了真实答案的完整证据。', '',
        '统计方式：先在每个请求内计算练习题比例，再对请求等权平均。'
        '每个请求的 vanilla 取当前训练使用的前 N 题，specific 使用对应 N 题；不重抽样、不挑结果。'
        '四档 cache 共用这些记录，去重后只计一次。使用生成失败时已有的 query fallback，另列排除 fallback 的敏感性结果。', '',
        '| Dataset | 请求数 | 每套匹配练习次数 | Vanilla 聚焦率 ↑ | Specific 聚焦率 ↑ | Vanilla 非相邻材料同现 ↑ | Specific 非相邻材料同现 ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    table = []
    for d in DATASETS:
        x = summary[d]; a=x['arms']['vanilla']; b=x['arms']['specific']
        lines.append(f"| {NAMES[d]} | {x['requests']} | {x['matched_exercise_exposures_per_arm']} | "
            f"{a['query_term_focus']:.2f}% | {b['query_term_focus']:.2f}% | "
            f"{a['disjoint_source_context']:.2f}% | {b['disjoint_source_context']:.2f}% |")
        for metric in METRICS:
            table.append(dict(dataset=d, metric=metric, requests=x['requests'],
                exercise_exposures_per_arm=x['matched_exercise_exposures_per_arm'],
                vanilla_percent=a[metric], specific_percent=b[metric],
                difference_percentage_points=b[metric]-a[metric]))
    lines += ['', '## 计数与限制', '',
        '这是当前三个数据集的 **107 个请求、每套 642 次练习暴露**，不是早先的 1,000 题 BABILong。'
        'Vanilla 的同一份文档级题库会被多个请求复用，642 不表示新生成了 642 个不同问题。', '',
        '| Dataset | 不同原文数 | 使用 query fallback 的请求 | Vanilla 不同记录数 | Specific 不同记录数 |',
        '|---|---:|---:|---:|---:|']
    for d in DATASETS:
        x=summary[d]
        lines.append(f"| {NAMES[d]} | {x['documents']} | {x['query_fallback_requests']} | "
            f"{x['arms']['vanilla']['unique_source_records']} | {x['arms']['specific']['unique_source_records']} |")
    lines += ['', 'Vanilla 的非相邻材料同现为 0，是此批记录采用连续原文窗口的结果；'
        '它不意味着 Vanilla 没有跨块推理能力。当前图只支持“词面聚焦更多、非相邻材料共同呈现更多”的结论，'
        '不支持“正确证据更完整、问题更正确或下游准确率更高”的结论。', '',
        '## 排除使用 query fallback 的请求（配对敏感性分析）', '',
        '| Dataset | 请求数 | Vanilla 聚焦率 | Specific 聚焦率 | Vanilla 非相邻材料同现 | Specific 非相邻材料同现 |',
        '|---|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        a=summary[d]['arms']['vanilla']['excluding_query_fallback']
        b=summary[d]['arms']['specific']['excluding_query_fallback']
        lines.append(f"| {NAMES[d]} | {a['requests']} | {a['query_term_focus']:.2f}% | {b['query_term_focus']:.2f}% | "
            f"{a['disjoint_source_context']:.2f}% | {b['disjoint_source_context']:.2f}% |")
    lines += ['', '## 文件与复算', '',
        '- `summary.json`：汇总指标和排除 fallback 的敏感性统计。',
        '- `table.csv`：图中 6 个方法对照单元。',
        '- `per_request.csv`：全部请求级比例。',
        '- `records.jsonl`：每道练习的问句、实际命中词、原文坐标和 fallback 标记。',
        '- `manifest.json`：源文件 SHA256、分析代码与环境。',
        '- PDF/SVG 为矢量图；PNG 为 2240 × 2240 的正方形图。', '',
        '复算命令：', '', '```bash',
        'env PYTHONPATH=/home/da839/Query_Cart/.deps /home/da839/.conda/envs/TTT/bin/python \\',
        f"  {Path(__file__).resolve()} \\",
        f"  --campaign {manifest['campaign']} \\",
        '  --output /path/to/a/new/output_directory', '```', '',
        '## Figure caption', '',
        'Matched synthetic-record diagnostics for 107 requests from LongHealth, MTOB (KE), and QASPER. '
        'Blue and orange denote the query-agnostic and query-specific datasets used by Vanilla and Contextual '
        'Cartridges, respectively. Panel (a) shows the proportion of exercise questions sharing at least one '
        'non-stopword with the real query. Panel (b) shows the proportion of exercises whose attached original '
        'contexts contain at least two disjoint, non-adjacent source intervals. Values are unweighted means of '
        'within-request proportions under matched exercise counts. Both are record-level diagnostics; neither '
        'verifies semantic relevance, answer correctness, complete supporting evidence, or reasoning necessity. '
        'The zero Vanilla source-co-presentation rates follow from this run\'s contiguous-window construction. '
        'No SD outcomes or confidence intervals are included.', '',
        '## Software acknowledgement', '',
        'Figure preparation used the scientific-visualization skill from Scientific Agent Skills. '
        'Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026). '
        '[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents]'
        '(https://doi.org/10.48550/arXiv.2609.00065). Live arXiv record checked on 2026-11-12 (v2).',
    ]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    csv_write(output/'table.csv', table)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, case_rows, inputs = collect(args.campaign)
    summary=summarize(case_rows, records)
    with (args.output/'records.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    csv_write(args.output/'per_request.csv', case_rows)
    dump(args.output/'summary.json', summary)
    export=plot(summary, args.output)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(), campaign=str(args.campaign),
        source_json_sha256=inputs, plan_sha256=sha(args.campaign/'plan.json'),
        code=str(Path(__file__).resolve()), code_sha256=sha(__file__),
        query_term_code_sha256=sha(ROOT/'experiments/analyze_synthetic_comparison.py'),
        requests=len(case_rows)//2, exercise_exposures=len(records),
        source_coordinates_verified=True, inference_calls=0, new_gpu_jobs=0,
        training_modified=False, benchmark_answers_used_for_metrics=False,
        semantic_relevance_or_joint_reasoning_verified=False, export=export)
    dump(args.output/'manifest.json', manifest)
    report(summary, args.output, manifest)
    print(json.dumps({'output':str(args.output), 'summary':summary, 'export':export}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
