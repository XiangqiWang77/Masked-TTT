"""Two square ED-JSD diagnostic figures from a fixed, paired saved cohort.

No inference, training, checkpoint selection, or scheduler operations. The
contours describe the training-example pool at the shared warmup checkpoint,
not a before/after evaluation of final token distributions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil


SOURCE = Path('/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/'
              'contextual_ed_jsd_devel6_20261111_v1/babilong')
REFERENCES = Path('/home/da839/scratch_pi_mg269/da839/stack_ttt/'
                  'chunked_ntp_query_only_20261029/evaluation_references.json')
SCORER = Path('/home/da839/Stack_TTT/third_party/qa_suite/babilong_metrics.py')
DATASETS = ('vanilla_syns', 'specific_syns')
BLUE, ORANGE, INK = '#3180A8', '#E36A1C', '#0B2A3A'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(source, output):
    provenance = {}

    def read(path):
        raw = path.read_bytes()
        provenance[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    references = read(REFERENCES)
    provenance[str(SCORER)] = hashlib.sha256(SCORER.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('saved_babilong_metric', SCORER)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    config = read(source / 'config.json')
    trajectories, cases, paired_checks = [], [], []
    raw_rows = []
    for path in sorted((source / 'results/cases').glob('*.json')):
        result = read(path)
        if result['status'] != 'completed':
            raise ValueError('Unexpected non-complete aggregate result')
        case_id = result['case']
        if result['protocol'] != 'babilong-entropy-directed-coarsened-jsd-v1':
            raise ValueError('Do not mix selectors or experimental protocols')
        inputs = read(source / 'cases' / case_id / 'input.json')
        baselines = read(source / 'cases' / case_id / 'baselines.json')

        def score(prediction):
            return int(any(metric.compare_answers(answer, prediction, result['query'],
                                                  metric.TASK_LABELS[result['task']])
                           for answer in references[case_id]['answers']))

        for dataset in DATASETS:
            vanilla, selective = [result['arms'][dataset + '_' + method]
                                  for method in ('vanilla_sd', 'selective_sd')]
            vt, st = vanilla['training'], selective['training']
            audit = st['ed_jsd']
            assert vanilla['teacher_targets_sha256'] == selective['teacher_targets_sha256']
            assert vt['updates'] == st['updates']
            assert vanilla['training_packing'] == selective['training_packing']
            assert vt['train_conversations'] == st['train_conversations']
            h = st['warmup_steps']
            vh = {row['step']: row for row in vt['history']}
            sh = {row['step']: row for row in st['history']}
            warmup_error = max(abs(vh[step]['train_kd_loss'] - sh[step]['train_kd_loss'])
                               for step in range(1, h + 1))
            assert warmup_error < 1e-5
            paired_checks.append(dict(case=case_id, dataset=dataset,
                target_sha256=vanilla['teacher_targets_sha256'], updates=vt['updates'],
                warmup_steps=h, max_warmup_loss_difference=warmup_error,
                packing_matches=True, same_per_case_initialization=True,
                initialization_seed=inputs['seed']))
            candidate_rows = audit['trajectories']
            assert len(candidate_rows) == st['train_conversations']
            assert audit['kept'] == sum(bool(row['keep']) for row in candidate_rows)
            for row in candidate_rows:
                # This historical BABILong run scored every trajectory, without
                # the later structural source gate or absolute-confidence guard.
                assert row.get('scored', True)
                assert row['keep'] == (row['score'] > 0)
                assert abs(row['score']) <= row['mean_jsd_bits'] + 1e-10
                assert 0 <= row['mean_jsd_bits'] <= 1
                exercise = inputs['datasets'][dataset]['train'][row['index']]
                actual_keep = bool(row['keep'] or st.get('all_rejected_original_SD_fallback'))
                flat = dict(case=case_id, dataset=dataset, task=result['task'],
                    recorded_analysis_split=result.get('analysis_split', 'unspecified'),
                    document_tokens=result['length'], trajectory_index=row['index'],
                    keep=bool(row['keep']), actual_consolidation_keep=actual_keep,
                    score=row['score'], mean_jsd_bits=row['mean_jsd_bits'],
                    teacher_entropy_bits=row['mean_teacher_bucket_entropy_bits'],
                    student_entropy_bits=row['mean_student_bucket_entropy_bits'],
                    tokens=row['tokens'], positive_tokens=row['positive_tokens'],
                    negative_tokens=row['negative_tokens'],
                    teacher_rest_mass=row['mean_teacher_rest_mass'],
                    student_rest_mass=row['mean_student_rest_mass'],
                    question=exercise['question'], synthetic_answer=exercise.get('answer', ''),
                    answer_finished=exercise.get('answer_finished'),
                    evidence_sha256=hashlib.sha256(exercise['evidence'].encode()).hexdigest())
                trajectories.append(flat)
                raw_rows.append(dict(case=case_id, dataset=dataset, gate_record=row,
                                     question=exercise['question'],
                                     synthetic_answer=exercise.get('answer', '')))
            cases.append(dict(case=case_id, dataset=dataset, query=result['query'],
                recorded_analysis_split=result.get('analysis_split', 'unspecified'),
                task=result['task'], document_tokens=result['length'],
                reference=' | '.join(references[case_id]['answers']),
                SD_prediction=vanilla['prediction'], JSD_prediction=selective['prediction'],
                SD_correct=score(vanilla['prediction']), JSD_correct=score(selective['prediction']),
                teacher_full_prediction=baselines.get('full_context_teacher'),
                teacher_full_correct=score(baselines['full_context_teacher'])
                    if isinstance(baselines.get('full_context_teacher'), str) else None,
                initial_prediction=baselines.get('initial_kv'),
                initial_correct=score(baselines['initial_kv'])
                    if isinstance(baselines.get('initial_kv'), str) else None,
                trajectories=audit['candidates'], kept=audit['kept'], blocked=audit['removed'],
                fallback=bool(st.get('all_rejected_original_SD_fallback')),
                updates=st['updates'], warmup_steps=h,
                same_teacher_targets=True, warmup_max_loss_error=warmup_error))

    # Include every complete aggregate case, regardless of outcome or selection.
    assert len(cases) == 2 * len({row['case'] for row in cases})
    save_csv(output / 'trajectory_metrics.csv', trajectories)
    save_csv(output / 'paired_case_results.csv', cases)
    save_csv(output / 'blocked_trajectories.csv', [r for r in trajectories if not r['keep']])
    with (output / 'raw_gate_records.jsonl').open('w') as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(output / 'paired_audit.json', paired_checks)
    write_json(output / 'source_sha256.json', provenance)
    write_json(output / 'run_configuration.json', {k: config.get(k) for k in (
        'protocol', 'model', 'prefix_tokens', 'packed_tokens', 'learning_rate',
        'official_epochs', 'teacher_topk', 'fixed_synthetic_answer_trajectories',
        'teacher_context_policy', 'question_budget_policy', 'seed')})
    shutil.copyfile(SCORER, output / 'babilong_metric.py')
    return trajectories, cases


def load_tables(output):
    with (output / 'trajectory_metrics.csv').open() as handle:
        trajectories = list(csv.DictReader(handle))
    numeric = ('score', 'mean_jsd_bits', 'teacher_entropy_bits', 'student_entropy_bits',
               'teacher_rest_mass', 'student_rest_mass')
    integer = ('trajectory_index', 'document_tokens', 'tokens', 'positive_tokens', 'negative_tokens')
    for row in trajectories:
        for key in numeric:
            row[key] = float(row[key])
        for key in integer:
            row[key] = int(row[key])
        for key in ('keep', 'actual_consolidation_keep'):
            row[key] = row[key] == 'True'
    with (output / 'paired_case_results.csv').open() as handle:
        cases = list(csv.DictReader(handle))
    for row in cases:
        for key in ('SD_correct', 'JSD_correct', 'trajectories', 'kept', 'blocked', 'updates',
                    'warmup_steps', 'document_tokens'):
            row[key] = int(row[key])
        for key in ('teacher_full_correct', 'initial_correct'):
            row[key] = int(row[key]) if row[key] else None
        row['fallback'] = row['fallback'] == 'True'
    return trajectories, cases


def summarize(trajectories, cases):
    summary = {}
    for dataset in DATASETS:
        ts = [r for r in trajectories if r['dataset'] == dataset]
        cs = [r for r in cases if r['dataset'] == dataset]
        transitions = {f'{a}_to_{b}': sum(r['SD_correct'] == a and r['JSD_correct'] == b for r in cs)
                       for a in (0, 1) for b in (0, 1)}
        by_split = {}
        for split in sorted({r['recorded_analysis_split'] for r in cs}):
            rows = [r for r in cs if r['recorded_analysis_split'] == split]
            by_split[split] = dict(n=len(rows),
                SD_correct=sum(r['SD_correct'] for r in rows),
                JSD_correct=sum(r['JSD_correct'] for r in rows),
                fixes=sum(r['SD_correct'] == 0 and r['JSD_correct'] == 1 for r in rows),
                regressions=sum(r['SD_correct'] == 1 and r['JSD_correct'] == 0 for r in rows))
        summary[dataset] = dict(requests=len(cs), trajectories=len(ts), by_recorded_split=by_split,
            kept=sum(r['keep'] for r in ts), blocked=sum(not r['keep'] for r in ts),
            gate_active_requests=sum(0 < r['kept'] < r['trajectories'] for r in cs),
            SD_correct=sum(r['SD_correct'] for r in cs), JSD_correct=sum(r['JSD_correct'] for r in cs),
            SD_accuracy=100 * sum(r['SD_correct'] for r in cs) / len(cs),
            JSD_accuracy=100 * sum(r['JSD_correct'] for r in cs) / len(cs),
            net_gain_pp=100 * sum(r['JSD_correct'] - r['SD_correct'] for r in cs) / len(cs),
            transitions=transitions, fallback_requests=sum(r['fallback'] for r in cs),
            mean_teacher_entropy_all=sum(r['teacher_entropy_bits'] for r in ts) / len(ts),
            mean_teacher_entropy_kept=sum(r['teacher_entropy_bits'] for r in ts if r['keep']) /
                                      sum(r['keep'] for r in ts),
            mean_entropy_sign_gate_disagreements=sum(
                (r['student_entropy_bits'] > r['teacher_entropy_bits']) != r['keep'] for r in ts),
            corrected_cases=[r['case'] for r in cs if r['SD_correct'] == 0 and r['JSD_correct'] == 1],
            regressed_cases=[r['case'] for r in cs if r['SD_correct'] == 1 and r['JSD_correct'] == 0],
            complete_case_ids=sorted(r['case'] for r in cs))
    return summary


def density_contours(points, mask, grid, bandwidth):
    """Shared diagonal Gaussian kernel with reflection at the zero boundaries.

    Every trajectory has equal weight. Each distribution integrates to one on
    the displayed nonnegative grid. Bandwidth is fitted once on the ALL pool.
    """
    import numpy as np
    xs, ys = grid
    chosen = points[mask]
    zx = np.zeros((len(xs), len(chosen)))
    zy = np.zeros((len(ys), len(chosen)))
    for sign in (-1, 1):
        zx += np.exp(-.5 * ((xs[:, None] - sign * chosen[None, :, 0]) / bandwidth[0]) ** 2)
        zy += np.exp(-.5 * ((ys[:, None] - sign * chosen[None, :, 1]) / bandwidth[1]) ** 2)
    z = zy @ zx.T / (len(chosen) * 2 * np.pi * bandwidth[0] * bandwidth[1])
    z /= z.sum() * (xs[1] - xs[0]) * (ys[1] - ys[0])
    ordered = np.sort(z.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    thresholds = [float(ordered[np.searchsorted(cumulative, p)]) for p in (.8, .5)]
    return z, thresholds


def draw(output, trajectories, cases, summary, *, distribution_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    import numpy as np

    ts = [r for r in trajectories if r['dataset'] == 'specific_syns']
    points = np.array([[r['teacher_entropy_bits'], r['student_entropy_bits']] for r in ts])
    keep = np.array([r['keep'] for r in ts])
    # Density contours are descriptive, not uncertainty/confidence intervals.
    bandwidth = points.std(axis=0, ddof=1) * len(points) ** (-1 / 6)
    limit = math.ceil((points.max() + .03) * 5) / 5
    grid = (np.linspace(0, limit, 220), np.linspace(0, limit, 220))
    density_records = {}
    rc = {'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9.5,
          'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
          'axes.edgecolor': INK, 'axes.linewidth': .7, 'pdf.fonttype': 42,
          'svg.fonttype': 'none', 'figure.facecolor': 'white', 'axes.facecolor': 'white'}

    def export(fig, stem):
        from matplotlib.text import Text
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        for artist in fig.findobj(Text):
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            box = artist.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > width + .5 or box.y1 > height + .5:
                raise ValueError(f'Text clipped in {stem}: {artist.get_text()}')
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output / (stem + '.' + ext), dpi=400, facecolor='white', transparent=False)
        plt.close(fig)

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(3.6, 3.6))
        ax = fig.add_axes([.18, .16, .76, .76])
        ax.set_aspect('equal', adjustable='box')
        # Deleted trajectories stay visible; no point is jittered or clipped.
        ax.scatter(points[keep, 0], points[keep, 1], s=6, c=ORANGE, alpha=.17,
                   edgecolors='none', zorder=1)
        ax.scatter(points[~keep, 0], points[~keep, 1], s=12, c=INK, alpha=.38,
                   marker='x', linewidths=.55, zorder=2)
        for label, mask, color, style in (
                ('SD', np.ones(len(points), dtype=bool), BLUE, '--'),
                ('JSD-SD', keep, ORANGE, '-')):
            z, levels = density_contours(points, mask, grid, bandwidth)
            ax.contour(*grid, z, levels=levels, colors=color,
                       linestyles=style, linewidths=[1.15, 1.7], zorder=3)
            density_records[label] = {'count': int(mask.sum()), 'levels': levels}
        ax.plot([0, limit], [0, limit], color='#72818A', lw=.8, ls=':', zorder=0)
        ax.set(xlim=(0, limit), ylim=(0, limit),
               xlabel='Teacher entropy (bits)', ylabel='Student entropy (bits)')
        ax.set_xticks([0, .4, .8, 1.2, 1.6])
        ax.set_yticks([0, .4, .8, 1.2, 1.6])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3, pad=3)
        ax.legend(handles=[
            Line2D([0], [0], color=BLUE, ls='--', lw=1.5, label='Vanilla SD'),
            Line2D([0], [0], color=ORANGE, lw=1.6, label='JSD-SD'),
            Line2D([0], [0], color=INK, marker='x', ls='none', markersize=4,
                   label='Blocked')],
            loc='upper left', frameon=False, fontsize=7.7, handlelength=2.2,
            labelspacing=.38, borderaxespad=.25)
        export(fig, 'jsd_training_distribution')
        if distribution_only:
            return

        fig = plt.figure(figsize=(3.6, 3.6))
        ax = fig.add_axes([.25, .19, .69, .69])
        ax.set_aspect('equal', adjustable='box')
        info = summary['specific_syns']
        counts = info['transitions']
        labels = {(0, 0): 'Still wrong', (0, 1): 'Corrected',
                  (1, 0): 'Regressed', (1, 1): 'Still correct'}
        for a in (0, 1):
            for b in (0, 1):
                n = counts[f'{a}_to_{b}']
                color = '#1E4B6B' if a == b else ORANGE if b > a else '#8C3A1A'
                if n:
                    # Circle AREA, not radius, is proportional to request count.
                    radius = .34 * math.sqrt(n / max(counts.values()))
                    ax.add_patch(Circle((a, b), radius, facecolor=color,
                                       edgecolor=INK, linewidth=.65, alpha=.93))
                ax.text(a, b, str(n), ha='center', va='center',
                        fontsize=12 if n else 11, fontweight='bold',
                        color='white' if n and a == b else INK)
                ax.text(a, b + .39, labels[a, b], ha='center', va='center', fontsize=8)
        ax.axhline(.5, color='#DFE6E9', lw=.7, zorder=0)
        ax.axvline(.5, color='#DFE6E9', lw=.7, zorder=0)
        ax.set(xlim=(-.5, 1.5), ylim=(-.5, 1.5),
               xlabel='Vanilla SD outcome', ylabel='JSD-SD outcome')
        ax.set_xticks([0, 1], ['Wrong', 'Correct'])
        ax.set_yticks([0, 1], ['Wrong', 'Correct'])
        ax.tick_params(length=0, pad=6)
        ax.spines[['top', 'right']].set_visible(False)
        export(fig, 'jsd_paired_outcomes')

    write_json(output / 'plot_specification.json', dict(
        source='Query-specific data, every complete saved BABILong factorial request',
        comparison='same synthetic questions / Teacher targets / initial KV / optimizer-step budget',
        figure_count=2, dimensions_inches=[3.6, 3.6], dpi=400,
        axes_entropy='trajectory means of top-k-plus-rest bucket entropies at warmup',
        diagonal='equal mean entropy, NOT the exact ED-JSD decision boundary',
        density_kernel='diagonal Gaussian, shared Scott bandwidth, reflected at zero',
        bandwidth_bits=bandwidth.tolist(), grid_shape=[220, 220],
        contour_enclosed_probability=[.8, .5],
        contours_are_confidence_intervals=False, density_normalization='one per pool',
        density_weighting='one equally weighted observation per synthetic trajectory',
        density_records=density_records,
        points='all retained observations as orange dots; all blocked as navy crosses',
        outcome_circle='area proportional to number of requests; zero has no circle',
        no_generated_or_interpolated_results=True, gold_used_only_for_final_scoring=True,
        known_confounds='Contour pool shift is the selection operation, not proof of causal correctness.'))


def describe(output, summary, trajectories, cases):
    s = summary['specific_syns']; a = summary['vanilla_syns']
    changed = [r for r in cases if r['dataset'] == 'specific_syns' and r['SD_correct'] != r['JSD_correct']]
    correction_lines = []
    for row in changed:
        # Keep the complete raw predictions in CSV. The display description
        # explicitly labels a malformed output without pretending it is an answer.
        vanilla = row['SD_prediction'] if len(row['SD_prediction']) < 35 else '乱码输出（原文见 CSV）'
        correction_lines.append('| ' + ' | '.join([
            row['case'], row['query'].strip(), row['reference'], vanilla,
            row['JSD_prediction'], f"{row['blocked']}/{row['trajectories']}"]) + ' |')
    readme = r'''# ED-JSD: what was filtered, and what changed?

两张独立正方形图，使用已保存的 **32 个 BABILong 请求、Qwen3-4B、512-slot KV**。
这是 2026-11-11 已停止实验的完整配对快照，**不是当前三数据集实验，也不是 1,000 请求的完整结果**。
没有重跑模型，没有根据结果选择样本。
原实验将其中 7 个请求标为 development、25 个标为 holdout；图中全部纳入，
不把总体 32 例当成独立、随机抽取的最终测试集。分组结果保留在 summary.json。

<img src="jsd_training_distribution.png" width="360" alt="Shared-warmup Teacher versus Student entropy contours for all 573 query-specific trajectories and the 390 retained by ED-JSD; 183 rejected trajectories remain visible as crosses.">

<img src="jsd_paired_outcomes.png" width="360" alt="Paired final outcomes on 32 requests: 24 wrong under both SD and JSD-SD, six correct under both, two corrected by JSD-SD, zero regressed.">

## Figure 1 — 筛掉什么

横轴和纵轴分别是同一热身 checkpoint 上、每条完整回答轨迹的 Teacher 与 Student 平均熵，单位 bit。
使用实验实际保存的 **Teacher top-k + rest 合并桶**分布，不称作完整词表熵。
蓝色虚线表示普通 SD 会使用的全部轨迹，橙色实线表示 JSD-SD 留下的轨迹；深蓝叉号逐条标出被屏蔽轨迹。
两组等高线使用相同的核与带宽，分别圈住各自平滑分布的 50% 和 80% 概率质量，**不是置信区间**。
散点不抖动、不删异常值；分布按轨迹等权，不把 573 条练习当成 573 个独立测试请求。

实际门控仍是实验中的 token 聚合公式：

$$
J_i=\frac{1}{|y_i|}\sum_t\operatorname{JSD}_2(p_T^{i,t},p_S^{i,t})
\frac{H_2(p_S^{i,t})-H_2(p_T^{i,t})}{H_2(p_S^{i,t})+H_2(p_T^{i,t})+10^{-12}},
\qquad m_i=\mathbf 1[J_i>0].
$$

**对角线只是“两个平均熵相等”，不是门控分界线。** 由于门控先在 token 上做 JSD 加权再平均，
63/573 条轨迹的保留决定与“直接比较两个平均熵”不同；图上全部保留这些反例。
这也说明不能用一条平均熵对角线替代实际代码。

## Figure 2 — 最终答题发生什么变化

横轴是 Vanilla SD 最终回答的对错，纵轴是 JSD-SD 的对错。圆面积按请求数成比例；0 不画圆。
使用相同 32 个请求、同一套 query-specific 题目、同一 Teacher target SHA256、相同 KV 初始化和训练更新数。
逐步核对了全部 32 对热身损失，最大允许偏差为 $10^{-5}$；所有配对均通过。
横纵轴不是模型置信度，也不是合成练习题的正确性标签。

| 数据 | 请求数 | 轨迹数 | JSD 屏蔽 | SD 正确 | JSD-SD 正确 | 错→对 | 对→错 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Query-agnostic | 32 | 573 | 1 | 7/32 (21.875%) | 7/32 (21.875%) | 0 | 0 |
| Query-specific | 32 | 573 | 183 | 6/32 (18.75%) | 8/32 (25.00%) | 2 | 0 |

Query-specific 下，JSD 对 SD 的观察增益是 **+6.25 个百分点**；query-agnostic 下没有观察到增益。
所有 32 个请求都进入图，没有按门控激活、是否纠错或 Teacher 是否答错筛选。
本次保留集合均非空，因此没有触发“全拒绝后仍继续全量 SD”的回退。
这批实验使用历史的 $J_i>0$ 门控，未使用后来新增的 Teacher 绝对置信度阈值；
不把这些观测作为尚未运行的新配置结果。

## 哪些请求、哪些题被屏蔽

全部被屏蔽轨迹见 [blocked_trajectories.csv](blocked_trajectories.csv)，包含请求 ID、练习问题、
原有合成答案、轨迹索引、JSD 分数和双方熵；其中 183 条来自 query-specific，1 条来自 query-agnostic。
该文件的“synthetic_answer”是预先生成的轨迹文本，**不是人为核定的正确标签**。

两条最终纠错请求如下，完整输出保留在 [paired_case_results.csv](paired_case_results.csv)。

| 请求 ID | 最终测试问题 | 参考答案 | SD 输出 | JSD-SD 输出 | 屏蔽轨迹 |
|---|---|---|---|---|---:|
''' + '\n'.join(correction_lines) + r'''

这两例代表不同现象：`babilong-32-qa5-300105` 的初始 KV 和保存的全文 Teacher 都答对了 Fred，
普通 SD 最终产生乱码，JSD-SD 保留了正确回答；`babilong-64-qa6-300147` 保存的全文 Teacher、初始 KV
和普通 SD 都输出 no，而 JSD-SD 输出正确的 yes。后者是这组记录中一例“超越保存的全文 Teacher”的观察。
全文 Teacher 输出由当时实验从旧记录复用，不把它包装为本次新跑的 Teacher 对照。

**屏蔽不等于判错。** 例如第一例中被屏蔽的练习题是 “Who received the apple from Mary?”，
其原有合成答案正是 “Fred”；其 $J=-0.069451$、Teacher 平均熵 0.7011 bit、Student 0.4689 bit。
该轨迹虽然被门控拦住，却不能据此称 Teacher 的答案错误。
第二例中也同时屏蔽了声称 John 回到花园的练习，以及把 John 混为无关 John Fleming 的练习。
因此，数据支持“门控改变后半程训练分布，并在这 32 例里带来两次最终纠错”，
**不支持“精确识别所有 Teacher 错误”“逐条删的都是错题”或“零退化保证”**。
没有做逐条轨迹删除的反事实消融，不能把某次纠错归因于单独一条被删题。

## 可复现资料

- [全部轨迹数值 CSV](trajectory_metrics.csv) / [原始门控记录 JSONL](raw_gate_records.jsonl)
- [全部配对最终输出 CSV](paired_case_results.csv) / [配对条件核查](paired_audit.json)
- [分组统计](summary.json) / [图形变换、核带宽与轮廓定义](plot_specification.json)
- [运行配置](run_configuration.json) / [原始文件 SHA256](source_sha256.json)
- [PNG/PDF/SVG 导出核查](export_audit.json) / [绘图脚本](reproduce.py)

```bash
python reproduce.py --output . --render-only
```

渲染只需 NumPy、Matplotlib 和 Pillow，不需要 GPU、模型或原始文档。
CSV 中的参考答案只用于这个离线结果分析，不参与 Teacher、训练、门控或 checkpoint 选择。
未生成的请求不补零、不外推；没有置信区间。默认原生大小为 3.6 × 3.6 inch，PNG 400 dpi，
并提供矢量 PDF/SVG。这是通用论文插图，不声称已通过某出版社格式认证。

## Claim–evidence audit

| Claim | Evidence | Status |
|---|---|---|
| Gate blocks a subset of the query-specific training pool | 183/573 saved decisions have J ≤ 0 | Supported |
| SD/JSD-SD use identical supervision within each pair | Target hashes, packing, update counts, common warmup traces | Supported |
| JSD improves this completed query-specific cohort | 6/32 → 8/32; 2 fixes, 0 regressions | Supported for these 32 requests |
| Removed trajectories are all wrong | Blocked answer “Fred” is a counterexample | Not supported |
| These figures establish a gain on three other datasets or all 1,000 requests | Those paired results are absent | Not supported |

Review: contribution is stated as observed selection behavior; the count rather than contour appearance supports
the performance claim; all completed cases and both data distributions are reported; missing work is not a zero;
confidence and correctness are separated. These checks do not establish generalization or calibration.

## Software acknowledgement

Figure preparation used Scientific Agent Skills. Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M.
(2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
[arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065), current record checked 2026-11-12.
'''
    (output / 'README.md').write_text(readme)
    (output / 'captions.txt').write_text(
        'Figure 1. Training-pool selection at the common warmup checkpoint. Blue dashed contours include '
        'all 573 query-specific trajectories; orange solid contours include the 390 retained by ED-JSD. '
        'Crosses show all 183 rejected trajectories. Coordinates are trajectory-mean top-k-plus-rest '
        'bucket entropies; the diagonal is equal mean entropy, not the gate boundary. Contours enclose '
        '50% and 80% of separately normalized, shared-bandwidth reflected KDEs; they are not confidence '
        'intervals. These are selection-pool distributions, not final model probability distributions.\n\n'
        'Figure 2. Paired final-answer transitions on the same 32 completed BABILong requests using '
        'query-specific data, matched Teacher targets, initial KV, and update budgets. The cohort '
        'contains seven recorded development and 25 recorded holdout requests. Circle area '
        'encodes count: 24 remain wrong, six remain correct, two are corrected, and none regress. '
        'Accuracy rises from 18.75% to 25.00%. This partial historical cohort does not establish a '
        'non-regression guarantee or a result for the three-dataset experiment.\n')


def audit_exports(output):
    from PIL import Image
    import matplotlib
    import numpy
    import PIL
    import xml.etree.ElementTree as ET
    exports = []
    for name in ('jsd_training_distribution', 'jsd_paired_outcomes'):
        with Image.open(output / (name + '.png')) as img:
            assert img.size == (1440, 1440)
            assert img.convert('RGBA').getchannel('A').getextrema() == (255, 255)
            exports.append(dict(file=name+'.png', size_pixels=list(img.size),
                dpi=list(img.info['dpi']), opaque=True))
        svg = ET.parse(output / (name + '.svg')).getroot()
        assert svg.attrib['width'] == svg.attrib['height'] == '259.2pt'
    write_json(output / 'export_audit.json', dict(
        created=datetime.now().astimezone().isoformat(), exports=exports,
        packages={'numpy':numpy.__version__, 'matplotlib':matplotlib.__version__, 'pillow':PIL.__version__},
        physical_square=True, PDF_fonttype=42, SVG_text_editable=True,
        confidence_intervals=False, raw_source_files_modified=False,
        no_model_or_gpu_execution=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--render-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        trajectories, cases = load_tables(args.output)
    else:
        trajectories, cases = extract(args.source, args.output)
    summary = summarize(trajectories, cases)
    write_json(args.output / 'summary.json', summary)
    draw(args.output, trajectories, cases, summary)
    describe(args.output, summary, trajectories, cases)
    audit_exports(args.output)
    destination = args.output / 'reproduce.py'
    if destination.resolve() != Path(__file__).resolve():
        shutil.copyfile(__file__, destination)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='complete_case_ids'}
                      for k,v in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
