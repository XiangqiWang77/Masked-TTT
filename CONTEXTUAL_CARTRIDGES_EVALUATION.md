# Contextual Cartridges：冻结评估协议

状态：2026-11-09。Query-Specific v3 合成数据已经完成；四臂原生 KV 开发实验正在运行。本文档只定义如何验证两项技术，不把未完成准确率写成结果。

## 1. 数据与任务

完整数据覆盖 BABILong 的 1,000 个请求、10 个任务类型与四种长度：8K、16K、32K、64K 各 250 题。Query-Specific v3 包含 28,671 个 `(question, answer anchor, evidence)` 三元组，其中 21,207 个 train、7,464 个 validation。Vanilla synthetic 来自官方 Cartridges 风格的 query-agnostic 生成。四臂实验对每个 case 取相同的 deterministic common-prefix 数量，避免用更多问题数量解释方法差异。

Answer-first 的 $a_i$ 只用于反推练习题，并不是 Student 的 hard label。Student 始终学习来源条件下 Teacher 的 sparse soft distribution。benchmark 隐藏答案不进入数据生成、Teacher 选择、TracIn 或 KV 更新。

## 2. 四臂因子设计

| Synthetic questions | Learning rule | 实验臂 | 隔离的变量 |
| --- | --- | --- | --- |
| Query-agnostic | Vanilla SD | `vanilla_syns_vanilla_sd` | 原版控制 |
| Query-agnostic | TracIn-Selective SD | `vanilla_syns_selective_sd` | 固定数据验证 Selective SD |
| Query-specific Answer-Guided | Vanilla SD | `specific_syns_vanilla_sd` | 固定 SD 验证问题构造 |
| Query-specific Answer-Guided | TracIn-Selective SD | `specific_syns_selective_sd` | 完整 Contextual Cartridges |

四臂从同一 per-case 512-token native KV 初始化出发，Backbone 冻结，使用原版 sparse Teacher KL、2,048-token packing 与相同的优化器设置。Vanilla 控制决定实际 packed optimizer-update 数；其余臂匹配这个预算。最终推理不读取原文，只读取训练后的 Cartridge 与真实问题。

## 3. Selective SD 的冻结规则

Selective SD 不使用 held-out label。它先用保守来源规则形成高召回候选池：明确被来源反驳、明确没有答案或截断的 Teacher 轨迹不作为选择性巩固候选，不确定轨迹保留。然后在初始 KV 与训练后 Vanilla-SD KV 两个检查点上计算 leave-one-out signed TracIn：

$$
I_i=\sum_c\eta_c\,\nabla_Z\mathcal L_{\mathrm{ref}\setminus i}(Z_c)^\top
\nabla_Z\mathcal L_i(Z_c).
$$

固定阈值为零，只保留 $I_i>0$。每个 Selective 臂先在全部普通 SD 轨迹上完成前半预算，再用保留轨迹完成后半 consolidation。若一个 case 没有正影响候选，直接复用已经训练好的同数据 Vanilla-SD checkpoint；不会返回未训练初始化，也不会通过另一随机顺序重跑 Vanilla 制造虚假差异。

TracIn 是一阶局部 influence proxy，不是证据真实性或最终 QA 非退化保证。最终报告必须保留所有 regressions。

## 4. 两项贡献的判定

### 4.1 Dilution 与 Fragmentation

数据层报告以下代理，不把它们等同于答题正确率：

- 真实查询词命中率；
- 支持事实召回下界；
- 完整支持链同上下文曝光下界；
- 多块证据比例；
- 问题重复、答案多样性、数据短缺与生成时间。

功能层用 `specific_syns_vanilla_sd − vanilla_syns_vanilla_sd` 判定 Query-Specific 数据在相同 SD 下是否改善最终回答。跨事实题与高干扰长文应单独报告。

### 4.2 Oracle Trap

固定同一数据，分别计算 Selective 相对 Vanilla 的：

- overall accuracy delta；
- paired fixes：Vanilla 错、Selective 对；
- paired regressions：Vanilla 对、Selective 错；
- net fixes minus regressions；
- `full-context Teacher wrong ∧ initial KV wrong` 严格 cohort 中的新增修复与新增退化。

只有严格 cohort 的净修复为正，才支持“方法有机会越过无条件 Teacher 模仿”的主张。Teacher-wrong/Student-correct 但 initial KV 已经正确的 case 不算新学到的修复。

## 5. Development gate 与 holdout

冻结的 source-aware 划分包含：

| Partition | Cases | Source questions | 用途 |
| --- | ---: | ---: | --- |
| Development | 160 | 40 | 验证实现、决定是否打开 holdout |
| Holdout | 840 | 210 | 开发规则冻结后的确认 |

只有 160-case development 同时满足以下条件，才允许打开 840-case holdout：

1. Query-Specific + Vanilla SD 优于 Query-Agnostic + Vanilla SD；
2. Selective SD 在两种固定数据上均优于对应 Vanilla SD；
3. 完整方法优于 Vanilla Cartridges；
4. 严格 Oracle-escape cohort 中 fixes 减 regressions 为正；
5. 四臂更新预算与数据计数通过审计。

开发门未通过时不得通过反复查看 holdout 来调阈值；阈值 $\tau=0$、两检查点与两阶段 schedule 已冻结。

## 6. 评分、缺失值与图表

准确率使用未修改的 BABILong evaluator。所有 delta 都使用同一批四臂完整 case；缺失预测记为 pending/NA，而不是错误或零分。最终图只在 160/160 完整且没有 report issue 时生成，并同时输出 PDF、SVG、PNG、LaTeX 表与源数据 CSV。图表不显示未经定义的误差条；当前 development cohort 是一个固定样本集，不伪装成多随机种子实验。

最终报告至少包含：

1. 四臂总体准确率；
2. 8K/16K/32K/64K 分长度准确率；
3. 四个关键对比的 fixes 与 regressions；
4. 两种数据下严格 Oracle-escape fixes/regressions；
5. 两检查点正 influence 比例与 TracIn keep/delete 比例；
6. 每臂实际更新次数、训练会话数、fallback 数和端到端时间。

## 7. 可复现文件

- [完整方法](CONTEXTUAL_CARTRIDGES_METHOD.zh-CN.md)
- [可读协议配置](datasets/babilong_synthetic_qwen3_4b/sd_protocol/configuration.json)
- [精确集群配置](datasets/babilong_synthetic_qwen3_4b/sd_protocol/configuration.cluster.json)
- [TracIn 实现](datasets/babilong_synthetic_qwen3_4b/sd_protocol/code/contextualized/tracin_sd.py)
- [四臂 runner](datasets/babilong_synthetic_qwen3_4b/sd_protocol/code/experiments/run_syn_sd_factorial.py)
- [严格配对 report](datasets/babilong_synthetic_qwen3_4b/sd_protocol/code/experiments/report_syn_sd_factorial.py)
- [完整结果绘图器](datasets/babilong_synthetic_qwen3_4b/sd_protocol/code/experiments/plot_tracin_sd_factorial.py)
