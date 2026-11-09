"""Separate dilution/fragmentation diagnostics for the full matched 1000-case data."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys


SEED = 20261109
METRICS = {
    'dilution_no_support': dict(field='no_support_context_fraction', percent=True, geometry=True,
        population='all', direction=-1, label='No-support exercise contexts', zh='无支持材料占比 ↓'),
    'dilution_query_focus': dict(field='query_term_hit_proxy', percent=True, geometry=False,
        population='all', direction=1, label='Query-term focus proxy', zh='问题关键词聚焦率 ↑'),
    'support_recall': dict(field='context_union_support_recall_min', percent=True, geometry=True,
        population='all', direction=1, label='Dataset support recall', zh='整套材料支持事实召回率 ↑'),
    'fragmentation_joint_exercises': dict(field='complete_context_fraction_min', percent=True, geometry=True,
        population='cross_chunk', direction=1, label='Joint-support exercise contexts', zh='跨块完整证据练习题占比 ↑'),
    'fragmentation_joint_cases': dict(field='has_complete_context_min', percent=True, geometry=True,
        population='cross_chunk', direction=1, label='Cases with a joint-support exercise', zh='至少一道跨块完整证据题的样本占比 ↑'),
    'all_complete_contexts': dict(field='complete_context_fraction_min', percent=True, geometry=True,
        population='all', direction=1, label='All-support exercise contexts (all cases)', zh='全体样本完整证据练习题占比 ↑'),
    'query_copies': dict(field='exact_query_copy_fraction', percent=True, geometry=False,
        population='all', direction=0, label='Exact copies of the real query', zh='真实问题直接复制率'),
    'duplicates': dict(field='exact_duplicate_fraction', percent=True, geometry=False,
        population='all', direction=-1, label='Within-case exact duplicate questions', zh='样本内重复题占比 ↓'),
    'question_words': dict(field='mean_question_words', percent=False, geometry=False,
        population='all', direction=0, label='Words per question', zh='每道练习题词数'),
    'evidence_characters': dict(field='mean_evidence_characters', percent=False, geometry=True,
        population='all', direction=0, label='Source characters per exercise', zh='每道练习题材料字符数'),
}
MAIN = ('dilution_no_support','dilution_query_focus','support_recall',
        'fragmentation_joint_exercises','fragmentation_joint_cases')
ARMS = ('original','answer_guided')
COLORS = {'original':'#667085','answer_guided':'#0072B2'}
LABELS = {'original':'Query-agnostic','answer_guided':'Question-guided'}


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def eligible(row,spec):
    if spec['population']=='cross_chunk' and row['support_requires_multiple_original_chunks'] is not True:
        return False
    for arm in ARMS:
        value=row['arms'].get(arm)
        if value is None or value.get(spec['field']) is None:return False
        if spec['geometry'] and value['context_position_known_questions']!=value['questions']:return False
    return True


def estimate(rows,spec,replicates):
    import numpy as np
    rows=[row for row in rows if eligible(row,spec)]
    if not rows:return {'n':0,'source_questions':0,'episode_clusters':0,
        'original':None,'guided':None,'delta':None,'delta_ci95':[None,None],
        'original_ci95':[None,None],'guided_ci95':[None,None],
        'improved':0,'tied':0,'regressed':0}
    scale=100. if spec['percent'] else 1.
    a=np.array([float(row['arms']['original'][spec['field']]) for row in rows])*scale
    b=np.array([float(row['arms']['answer_guided'][spec['field']]) for row in rows])*scale
    strata=defaultdict(lambda:defaultdict(list))
    for index,row in enumerate(rows):
        cluster=row.get('episode_cluster') or row['sampling_unit']
        strata[row['task']][cluster].append(index)
    rng=np.random.default_rng(SEED)
    sums=np.zeros((replicates,2));counts=np.zeros(replicates)
    clusters=0
    for task in sorted(strata):
        members=list(strata[task].values());clusters+=len(members)
        values=np.array([[a[ix].sum(),b[ix].sum(),len(ix)] for ix in members])
        draw=rng.integers(0,len(values),size=(replicates,len(values)))
        resampled=values[draw].sum(axis=1)
        sums+=resampled[:,:2];counts+=resampled[:,2]
    boot=sums/counts[:,None]
    quantile=lambda values:list(map(float,np.quantile(values,[.025,.975])))
    difference=b-a;direction=spec['direction']
    return {'n':len(rows),'source_questions':len({row['sampling_unit'] for row in rows}),
        'episode_clusters':clusters,'original':float(a.mean()),'guided':float(b.mean()),
        'delta':float(difference.mean()),'original_ci95':quantile(boot[:,0]),
        'guided_ci95':quantile(boot[:,1]),'delta_ci95':quantile(boot[:,1]-boot[:,0]),
        'relative_change_percent':float(100*(b.mean()/a.mean()-1)) if a.mean() else None,
        'improved':int((direction*difference>1e-10).sum()) if direction else None,
        'tied':int((abs(difference)<=1e-10).sum()),
        'regressed':int((direction*difference < -1e-10).sum()) if direction else None}


def build_statistics(raw,replicates):
    rows=raw['cases']
    if len(rows)!=1000 or raw['overall']['paired_cases']!=1000 or not raw['question_budgets_matched_per_case']:
        raise ValueError('This report requires all 1000 same-case matched-count datasets')
    groups={'overall':rows}
    groups.update({f'length_{length}':[r for r in rows if r['length']==length]
                   for length in (8192,16384,32768,65536)})
    groups.update({f'task_qa{task}':[r for r in rows if r['task']==f'qa{task}'] for task in range(1,11)})
    groups.update({f'cell_qa{task}_{length}':[r for r in rows if r['task']==f'qa{task}' and r['length']==length]
                   for task in range(1,11) for length in (8192,16384,32768,65536)})
    groups.update({f'provenance_{kind}':[r for r in rows if r['provenance_group']==kind]
                   for kind in ('fresh','inherited')})
    values={name:{'cases':len(part),'metrics':{metric:estimate(part,spec,replicates)
                                            for metric,spec in METRICS.items()}}
            for name,part in groups.items()}
    excluded=[r['case'] for r in rows if not eligible(r,METRICS['dilution_no_support'])]
    supported=[r for r in rows if eligible(r,METRICS['support_recall'])
               and r['arms']['original']['context_union_support_recall_min']==1]
    retained=[r for r in supported if r['arms']['answer_guided']['context_union_support_recall_min']==1]
    return {'created_utc':datetime.now(timezone.utc).isoformat(),
        'schema':'matched-synthetic-dilution-fragmentation-v1',
        'benchmark_cases':1000,'unique_source_questions':250,
        'same_per_case_train_and_validation_counts':True,
        'train_questions_per_method':20558,'validation_questions_per_method':7391,
        'total_questions_per_method':27949,'backbone':'Qwen3-4B-Instruct-2507',
        'metric_specs':METRICS,'groups':values,
        'source_geometry_excluded_cases':excluded,
        'retention_audit':{'original_full_support_cases':len(supported),
            'guided_retains_full_support':len(retained),'guided_loses_some_support':len(supported)-len(retained),
            'retention_percent':100*len(retained)/len(supported) if supported else None,
            'lost_case_ids':[r['case'] for r in supported if r not in retained]},
        'bootstrap':{'replicates':replicates,'seed':SEED,'interval':'95% percentile',
            'unit':'original source episode cluster; preserve all repeated document lengths',
            'stratification':'task type; resample episode clusters within each task',
            'estimator':'equal-case macro mean; paired differences use the same cases'},
        'source_annotations_used_only_after_generation':True,
        'benchmark_gold_answer_values_used':False,'new_model_calls':0,'new_GPU_jobs':0,
        'semantic_correctness_or_reasoning_necessity_proven':False,
        'skill_software_reference':{
            'authors':['Timothy Kassis','Vinayak Agarwal','Yuhuan He','Darshil Patel','Aubrey M. Brueckner'],
            'year':2026,'title':'Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents',
            'doi':'10.48550/arXiv.2609.00065','record_checked_utc':'2026-11-09'}}


def csv_tables(report,output,raw):
    directory=output/'tables';directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for group,body in report['groups'].items():
        for metric,v in body['metrics'].items():
            records.append({'group':group,'metric':metric,'planned_cases':body['cases'],
                'valid_paired_cases':v['n'],'source_questions':v['source_questions'],
                'episode_clusters':v['episode_clusters'],
                'query_agnostic':v['original'],'question_guided':v['guided'],
                'guided_minus_agnostic':v['delta'],'delta_ci95_low':v['delta_ci95'][0],
                'delta_ci95_high':v['delta_ci95'][1],
                'unit':'percent / percentage points' if METRICS[metric]['percent'] else 'count',
                'improved_cases':v['improved'],'tied_cases':v['tied'],'regressed_cases':v['regressed']})
    for filename,chosen in [('all_metrics.csv',records),
        ('overall.csv',[r for r in records if r['group']=='overall']),
        ('by_length.csv',[r for r in records if r['group'].startswith('length_')]),
        ('by_task.csv',[r for r in records if r['group'].startswith('task_')]),
        ('by_task_length.csv',[r for r in records if r['group'].startswith('cell_')])]:
        with (directory/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(chosen[0]));writer.writeheader();writer.writerows(chosen)
    records=[]
    for row in raw['cases']:
        record={k:row[k] for k in ('case','task','length','sampling_unit','episode_cluster','provenance_group',
                                  'support_requires_multiple_original_chunks')}
        for arm in ARMS:
            for metric in METRICS:
                record[arm+'_'+metric]=row['arms'][arm][METRICS[metric]['field']]
        record['all_source_positions_known']=eligible(row,METRICS['dilution_no_support'])
        records.append(record)
    with (directory/'per_case_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (directory/'paired_transitions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','valid_paired_cases','improved','tied','regressed'])
        writer.writeheader()
        for key in MAIN:
            v=report['groups']['overall']['metrics'][key]
            writer.writerow({'metric':key,'valid_paired_cases':v['n'],
                'improved':v['improved'],'tied':v['tied'],'regressed':v['regressed']})


def cell(value):return '—' if value is None else f'{value:.2f}'


def metric_table(report,groups,metrics):
    lines=['| 分组 | 指标 | 有效配对样本 | Query-agnostic | Question-guided | Δ (guided−agnostic) | Δ 的95% CI |',
           '|---|---|---:|---:|---:|---:|---|']
    for label,key in groups:
        for metric in metrics:
            v=report['groups'][key]['metrics'][metric]
            delta='—' if v['delta'] is None else f'{v["delta"]:+.2f}'
            ci='—' if v['delta'] is None else f'[{v["delta_ci95"][0]:+.2f}, {v["delta_ci95"][1]:+.2f}]'
            lines.append(f'| {label} | {METRICS[metric]["zh"]} | {v["n"]} | '
                         f'{cell(v["original"])} | {cell(v["guided"])} | {delta} | {ci} |')
    return '\n'.join(lines)


def documents(report,output):
    overall=report['groups']['overall']['metrics']
    d=overall['dilution_no_support'];f=overall['fragmentation_joint_cases']
    text=['# Question-guided synthetic data：Dilution 与 Fragmentation 分别测量','',
        'Qwen3-4B-Instruct-2507；同一批1000个BABILong文档问题对；qa1–qa10；8K/16K/32K/64K各250题。'
        '两种方法逐题匹配训练与验证题数量，各20,558道训练题、7,391道验证题，共27,949道合成题。','',
        '本报告只比较合成数据，不在vanilla synthetic上运行SD。SD对照另在固定的question-guided数据上进行。','',
        f'主结果：无支持材料占比从{d["original"]:.2f}%降至{d["guided"]:.2f}%；'
        f'在{f["n"]}个确需跨原版chunk取证的样本上，至少一道完整证据练习题的覆盖率从'
        f'{f["original"]:.2f}%增至{f["guided"]:.2f}%。','',
        '![Dilution and fragmentation measured separately](figures/overview.png)','',
        '## 1. Dilution：练习预算有多少没有接触真实问题的支持事实？','',
        '对每个真实问题q，使用生成结束后读取的BABILong官方supporting-fact位置作为评估参照。'
        '“无支持材料占比”是合成题所绑定材料不包含任何一条q的支持事实的比例，越低越好。'
        '先在每个文档问题对内算占比，再对配对样本等权平均。它不是生成题错误率：未包含这些支持事实的题也可能提供其他有用的中间知识。','',
        '“问题关键词聚焦率”是合成问句命中真实问题非停用词的比例；只读问句，不把已知q前缀或生成答案计入。'
        '它是可复现的词面代理，不是语义相关性的人工评分。'
        '“整套材料支持事实召回率”衡量所有练习材料的并集能覆盖多少q的官方支持事实。','',
        metric_table(report,[('总体','overall')],MAIN[:3]),'',
        '## 2. Fragmentation：分散的支持事实有没有在同一道练习中聚合？','',
        '只纳入至少两条必要支持事实、且无法在同一个实际原版2048-token生成chunk内完整获得的样本。'
        '“跨块完整证据练习题占比”是其绑定材料同时覆盖q全部支持事实的练习题比例；'
        '“至少一道跨块完整证据题的样本占比”衡量每份合成数据是否至少有一次完整证据同现。','',
        '原版在这个子集上的0%由被评估的单chunk数据协议与子集定义共同决定；'
        '它不表示原版Cartridges模型的准确率为0，也不表示官方方法的所有配置都不能跨块推理。'
        '材料共同出现仍不等于问句必须联合这些材料才能回答。','',
        metric_table(report,[('跨块支持子集','overall')],MAIN[3:]),'',
        '## 3. 按文档长度','',
        '![Metrics by document length](figures/by_length.png)','',
        metric_table(report,[(name,f'length_{n}') for name,n in [('8K',8192),('16K',16384),('32K',32768),('64K',65536)]],
                     ['dilution_no_support','support_recall','fragmentation_joint_exercises','fragmentation_joint_cases']),'',
        '## 4. 按任务类型','',
        metric_table(report,[(f'qa{i}',f'task_qa{i}') for i in range(1,11)],
                     ['dilution_no_support','fragmentation_joint_exercises','fragmentation_joint_cases']),'',
        '## 5. 任务×长度：全部40个单元格','',
        '![Benefits by task and document length](figures/task_length_gains.png)','',
        '空白子集用“—”表示该单元格没有满足跨块定义的样本，不作为0分处理；负结果完整保留。','',
        metric_table(report,[(f'qa{i}/{label}',f'cell_qa{i}_{length}') for i in range(1,11)
            for label,length in [('8K',8192),('16K',16384),('32K',32768),('64K',65536)]],
            ['dilution_no_support','fragmentation_joint_cases']),'',
        '## 6. 重复与题目长度等质量控制','',
        metric_table(report,[('总体','overall')],['query_copies','duplicates','question_words','evidence_characters']),'',
        'Question-guided问句整体更短；短本身不等于质量更高。材料字符数反映单题绑定材料大小，'
        '不等于端到端GPU时间。完整数据包含24份已有生成结果及976份新生成结果，下表分开报告。','',
        metric_table(report,[('976份新生成','provenance_fresh'),('24份已有','provenance_inherited')],MAIN),'',
        '## 7. 配对收益与退步：不只报告平均提升','',
        '| 指标（按有利方向） | 有效配对 | 改善 | 不变 | 退步 |','|---|---:|---:|---:|---:|']
    for key in MAIN:
        v=overall[key]
        text.append(f'| {METRICS[key]["zh"]} | {v["n"]} | {v["improved"]} | {v["tied"]} | {v["regressed"]} |')
    retention=report['retention_audit']
    text += ['',f'原版材料并集已经覆盖全部支持事实的{retention["original_full_support_cases"]}个样本中，'
        f'guided保留了{retention["guided_retains_full_support"]}个（{retention["retention_percent"]:.2f}%），'
        f'仍有{retention["guided_loses_some_support"]}个丢失了一部分支持。'
        '因此平均改善明显，但本轮数据尚不满足“原版好的样本一个都不能损害”的严格保证。','',
        '## 测量协议与边界','',
        f'- 全部1000对数据均已保存；{len(report["source_geometry_excluded_cases"])}个样本中，'
        '至少一个原版token chunk的解码文本无法与记录的原文字符区间完全对应，因此涉及全量材料定位的指标统一排除双方该样本。'
        '其余993对用于主要支持事实指标；不依赖原文定位的词面聚焦率使用全部1000对。',
        '- 跨块子集296个文档问题对；同一个源问题可在不同长度下重复出现。',
        '- 1000个文档问题对来自250个源问题/源episode簇；95% CI按task分层、对源episode簇做配对bootstrap，'
        f'保留其所有长度，{report["bootstrap"]["replicates"]}次，随机种子{SEED}。',
        '- 所有百分比为每个有效case内计算后等权宏平均，Δ单位为百分点；字符数和词数为原单位差。',
        '- 支持事实标注只在数据生成完成后读取，不回传给检索、出题或训练。没有新模型调用、GPU job或生成检查循环。',
        '- 这些结果支持“训练材料更聚焦、跨块证据组织更完整”。尚未证明合成答案的语义正确率、'
        '每道题的跨块推理必要性、最终Cartridge准确率或Oracle Trap的解决。',
        '- 本对照评估完整question-guided生成流程，材料召回与出题分布一起改变；'
        '不能据此单独归因于“把q加入prompt”或某一个子步骤。',
        '- 本次两批历史生成的GPU数量、并发和缓存不同，不据此报告公平加速比。',
        '- 图是通用研究报告版本，未声称满足某一会议最终投稿版式要求。','',
        '## 完整数据与复现','',
        '- [总体CSV](tables/overall.csv)；[按长度CSV](tables/by_length.csv)；'
        '[按任务CSV](tables/by_task.csv)；[全部40单元格CSV](tables/by_task_length.csv)。',
        '- [全部统计CSV](tables/all_metrics.csv)；[逐case指标](tables/per_case_metrics.csv)；[统计JSON](comparison.json)。',
        '- 每张图提供PNG、可编辑SVG和PDF；配对分析与绘图代码保存在本目录code中。'
        '可用`python code/plot_matched_synthetic_comparison.py --statistics comparison.json --output reproduced`重绘。',
        '- 完整合成JSON见[两套1000题数据](../README.md)。','',
        '## 软件说明与参考','',
        '图表采用scientific-visualization技能的样式、导出及完整性检查流程，并按research-paper-writing技能检查结论与证据的对应。',
        'Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). '
        '*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. '
        '[arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065). 官方记录核对日期：2026-11-09。','']
    (output/'README.md').write_text('\n'.join(text))
    (output/'captions.md').write_text(
        'Overview. Separate data diagnostics for dilution and fragmentation on the matched 1,000-case Qwen3-4B BABILong '
        'synthetic datasets (27,949 examples per method). No-support contexts use 993 cases with exact source geometry; '
        'lexical focus uses all 1,000. Joint-support metrics use only 296 cases whose official supporting facts span '
        'multiple actual original 2,048-token generation chunks. Values are case-level macro means. Error bars are '
        '95% task-stratified episode-cluster bootstrap intervals; all repeated document lengths stay together. '
        'Joint source exposure does not establish semantic correctness or necessary multi-hop reasoning.\n\n'
        'Length. No-support context rate, support recall, and cross-chunk joint-exercise case coverage across '
        '8K–64K; each metric displays its eligible case count. Same estimator and intervals as Overview.\n\n'
        'Task × length. Positive values always favor question-guided: agnostic minus guided for no-support '
        'context rate; guided minus agnostic for cross-chunk joint-exercise case coverage. Missing cross-chunk '
        'subsets are N/A, not zero. All 40 cells and any negative values are retained.\n')


def plots(report,output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    scripts=Path('/home/da839/.codex/skills/scientific-visualization/scripts')
    if scripts.is_dir():
        sys.path.insert(0,str(scripts))
        from style_presets import style_context
        from figure_export import export_figure
    else:
        # Standalone replay of the released statistics needs only Matplotlib.
        from contextlib import contextmanager
        @contextmanager
        def style_context(*args,**kwargs):
            with plt.rc_context({'axes.spines.top':False,'axes.spines.right':False,
                'axes.axisbelow':True,'legend.frameon':False,'pdf.fonttype':42,
                'ps.fonttype':42,'svg.fonttype':'none'}):yield
        def export_figure(fig,filename,*,formats,dpi,provenance,**kwargs):
            filename=Path(filename)
            for ext in formats:
                fig.savefig(filename.with_suffix('.'+ext),dpi=dpi,facecolor='white',transparent=False)
            write(filename.with_suffix('.export.json'),{'provenance':provenance,'standalone_replay':True})
    folder=output/'figures';folder.mkdir(parents=True,exist_ok=True)
    provenance={'source_data':'comparison.json; tables/per_case_metrics.csv',
        'estimator':'case macro mean','uncertainty':report['bootstrap'],
        'geometry_exclusions':report['source_geometry_excluded_cases'],
        'missing':'N/A where no eligible multi-chunk cases','publisher_status':'general report; requirements not asserted'}
    def save(fig,name):
        export_figure(fig,folder/name,formats=['png','svg','pdf'],dpi=300,overwrite=True,
            bbox_inches=None,provenance=provenance,write_manifest=True)
        plt.close(fig)
    with style_context('default',palette_name='okabe_ito_on_white'):
        with plt.rc_context({'font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.titlesize':9,
                             'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8}):
            fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),layout='constrained')
            keys=('dilution_no_support','dilution_query_focus','fragmentation_joint_exercises','fragmentation_joint_cases')
            for letter,ax,key in zip('abcd',axes.flat,keys):
                v=report['groups']['overall']['metrics'][key]
                values=[v['original'],v['guided']]
                low=[values[i]-v[arm+'_ci95'][0] for i,arm in enumerate(('original','guided'))]
                high=[v[arm+'_ci95'][1]-values[i] for i,arm in enumerate(('original','guided'))]
                bars=ax.bar([0,1],values,width=.58,color=[COLORS[a] for a in ARMS],
                    edgecolor='#27303F',linewidth=.5,yerr=np.maximum(0,[low,high]),
                    capsize=3,error_kw={'elinewidth':.9})
                bars[0].set_hatch('///')
                for i,value in enumerate(values):
                    ax.annotate(f'{value:.2f}',(i,max(value,v[('original','guided')[i]+'_ci95'][1])),
                        xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=9)
                ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Percent')
                ax.set_xticks([0,1],[LABELS[a] for a in ARMS]);ax.grid(axis='y',alpha=.22)
                mechanism='Dilution' if key.startswith('dilution') else 'Fragmentation'
                arrow='lower is better' if METRICS[key]['direction']<0 else 'higher is better'
                title={'dilution_no_support':'No-support contexts',
                    'dilution_query_focus':'Query-term focus proxy',
                    'fragmentation_joint_exercises':'Joint-support contexts',
                    'fragmentation_joint_cases':'Joint-support cases'}[key]
                ax.set_title(f'({letter}) {mechanism}: {title}\n{arrow}; n = {v["n"]}',loc='left',pad=9)
            fig.suptitle('Question-guided data: dilution and fragmentation measured separately',fontsize=10)
            save(fig,'overview')
            fig,axes=plt.subplots(1,3,figsize=(9.0,3.1),layout='constrained')
            lengths=(8192,16384,32768,65536)
            for ax,key,title in zip(axes,
                ('dilution_no_support','support_recall','fragmentation_joint_cases'),
                ('Dilution: no-support contexts ↓','Support recall ↑','Fragmentation: joint-support cases ↑')):
                for arm,name,marker,linestyle in [('original','original','s','--'),('answer_guided','guided','o','-')]:
                    vals=[report['groups'][f'length_{n}']['metrics'][key] for n in lengths]
                    ys=np.array([v[name] if v[name] is not None else np.nan for v in vals])
                    lo=np.array([v[name+'_ci95'][0] if v[name] is not None else np.nan for v in vals])
                    hi=np.array([v[name+'_ci95'][1] if v[name] is not None else np.nan for v in vals])
                    ax.errorbar(range(4),ys,yerr=np.maximum(0,np.stack([ys-lo,hi-ys])),
                        label=LABELS[arm],marker=marker,linestyle=linestyle,color=COLORS[arm],capsize=2)
                labels=[f'{n//1024}K\nn={report["groups"][f"length_{n}"]["metrics"][key]["n"]}' for n in lengths]
                ax.set_xticks(range(4),labels);ax.set_ylim(-2,103);ax.set_yticks([0,25,50,75,100])
                ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.22);ax.set_ylabel('Percent')
            axes[0].legend(loc='upper left',fontsize=7)
            save(fig,'by_length')
            fig,axes=plt.subplots(1,2,figsize=(7.8,4.6),layout='constrained')
            arrays=[]
            for key,sign in [('dilution_no_support',-1),('fragmentation_joint_cases',1)]:
                arrays.append(np.array([[np.nan if (v:=report['groups'][f'cell_qa{task}_{length}']['metrics'][key])['delta'] is None
                    else sign*v['delta'] for length in lengths] for task in range(1,11)]))
            limit=max(5.,float(np.ceil(max(np.nanmax(abs(x)) for x in arrays)/5)*5))
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
            cmap=plt.colormaps['RdBu'].copy();cmap.set_bad('#E4E7EC')
            for ax,values,title in zip(axes,arrays,('Dilution: no-support reduction','Fragmentation: joint-case gain')):
                image=ax.imshow(values,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
                ax.set_xticks(range(4),['8K','16K','32K','64K']);ax.set_yticks(range(10),[f'qa{i}' for i in range(1,11)])
                ax.set_title(title,loc='left');ax.set_xlabel('Document length')
                for i in range(10):
                    for j in range(4):
                        value=values[i,j]
                        color='white' if np.isfinite(value) and abs(value)>limit*.58 else '#101828'
                        ax.text(j,i,'N/A' if not np.isfinite(value) else f'{value:+.1f}',
                                ha='center',va='center',fontsize=8,color=color)
            fig.colorbar(image,ax=axes,label='Benefit for question-guided (percentage points)',shrink=.88,pad=.03)
            save(fig,'task_length_gains')


def latex_table(report,output):
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Matched Qwen3-4B BABILong synthetic-data diagnostics. Both methods contain 27,949 examples over 1,000 document--query pairs. Source-based metrics exclude seven cases with unmappable original token boundaries. Fragmentation uses the 296-case cross-chunk support subset.}',
        r'\label{tab:synthetic-dilution-fragmentation}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Metric & $n$ & Agnostic & Guided & $\Delta$ (pp) \\',r'\midrule']
    for key in MAIN:
        v=report['groups']['overall']['metrics'][key]
        lines.append(f'{METRICS[key]["label"]} & {v["n"]} & {v["original"]:.2f} & {v["guided"]:.2f} & {v["delta"]:+.2f} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (output/'tables/overall.tex').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw')
    p.add_argument('--statistics',help='Redraw figures from the released comparison.json; no private paths needed')
    p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=4000)
    a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if a.statistics:
        report=read(a.statistics)
        (output/'tables').mkdir(parents=True,exist_ok=True)
    elif a.raw:
        raw=read(a.raw);report=build_statistics(raw,a.bootstrap)
        csv_tables(report,output,raw)
    else:p.error('Provide --raw or --statistics')
    write(output/'comparison.json',report)
    documents(report,output);latex_table(report,output);plots(report,output)
    print(json.dumps({'output':str(output),'paired_cases':1000,'headline':report['groups']['overall']['metrics']},ensure_ascii=False))


if __name__=='__main__':main()
