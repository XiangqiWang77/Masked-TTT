# Contextual Cartridges：冻结评估协议

状态：2026-11-09。Query-Specific v3 合成数据已经完成；四臂原生 KV 开发实验正在运行。本文档只定义如何验证两项技术，不把未完成准确率写成结果。

## 1. 数据与任务

完整数据覆盖 BABILong 的 1,000 个请求、10 个任务类型与四种长度：8K、16K、32K、64K 各 250 题。Query-Specific v3 包含 28,671 个 `(question, answer anchor, evidence)` 三元组，其中 21,207 个 train、7,464 个 validation。Vanilla synthetic 来自官方 Cartridges 风格的 query-agnostic 生成。四臂实验对每个 case 取相同的 deterministic common-prefix 数量，避免用更多问题数量解释方法差异。

Answer-first 的 $a_i$ 只用于反推练习题，并不是 Student 的 hard label。Student 始终学习来源条件下 Teacher 的 sparse soft distribution。benchmark 隐藏答案不进入数据生成、Teacher 选择、门控统计或 KV 更新。

## 2. 四臂因子设计

| Synthetic questions | Learning rule | 实验臂 | 隔离的变量 |
| --- | --- | --- | --- |
| Query-agnostic | Vanilla SD | `vanilla_syns_vanilla_sd` | 原版控制 |
| Query-agnostic | ED-JSD-Selective SD | `vanilla_syns_selective_sd` | 固定数据验证 Selective SD |
| Query-specific Answer-Guided | Vanilla SD | `specific_syns_vanilla_sd` | 固定 SD 验证问题构造 |
| Query-specific Answer-Guided | ED-JSD-Selective SD | `specific_syns_selective_sd` | 完整 Contextual Cartridges |

四臂从同一 per-case 512-token native KV 初始化出发，Backbone 冻结，使用原版 sparse Teacher KL、2,048-token packing 与相同的优化器设置。Vanilla 控制决定实际 packed optimizer-update 数；其余臂匹配这个预算。最终推理不读取原文，只读取训练后的 Cartridge 与真实问题。

## 3. Selective SD 的冻结规则

Selective SD 不使用 held-out label，选择过程零反向传播。预算 $T$ 由 query-agnostic 普通 SD 的数据布局决定，两套数据共享。前 $h=\lfloor T/2\rfloor$ 步在全部轨迹上普通 SD 预热。候选池先经过记录在 Teacher 生成时的确定性 Source-Support Gate：空回答、截断、显式拒答的轨迹不参与打分，但仍参与预热和全量回退。随后在固定 KV 处对每条剩余候选做一次前向，使用与 Teacher 相同的回答前缀，把保留 top-k 概率与全词表归一化的 Student 概率各自补一个 rest bucket，得到 $(k{+}1)$ 桶粗化分布。以 2 为底、$M=\tfrac12(T+S)$，逐 token 计算

$$J=\Big[H(M)-\tfrac12\big(H(T)+H(S)\big)\Big]\times\frac{H(S)-H(T)}{H(S)+H(T)+10^{-12}},$$

轨迹取均值，严格 $J>0$ 进入后半程巩固，沿用同一 sampler 与 Adam 状态。若没有轨迹通过，继续原数据与原优化器，确定性前向下逐位等于跑满 $T$ 步的普通 SD；若全部通过同样保持原 sampler 不变。不会返回未训练初始化，也不会通过另一随机顺序重跑 Vanilla 制造虚假差异。

粗化分布的熵与 JSD 不是全词表统计量；该分数是分布层面的选择统计量，不是证据真实性或最终 QA 非退化保证。最终报告必须保留所有 regressions。

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
5. 门控候选数、keep/removed 数、source-gate 拒绝数与 exact-fallback 次数；
6. 每臂实际更新次数、训练会话数、fallback 数和端到端时间。

## 7. 可复现文件

- [完整方法](CONTEXTUAL_CARTRIDGES_METHOD.zh-CN.md)
- [英文门控技术说明](paper/contextual_cartridges/ED_JSD_SD.md)
- [ED-JSD 选择器实现](paper/contextual_cartridges/code/implementation/ed_jsd.py)
- [结构 Source-Support Gate](paper/contextual_cartridges/code/implementation/source_gate.py)
- [三数据集 cache runner](paper/contextual_cartridges/code/implementation/run_figure4_contextual.py)
- [配对重训 runner](paper/contextual_cartridges/code/implementation/run_ratio_retrain.py)
- [30 题配对合成管线](paper/contextual_cartridges/code/implementation/thirty_study_data.py)
- [实现清单与哈希](paper/contextual_cartridges/data/implementation_manifest.json)
