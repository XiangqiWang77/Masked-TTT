# Query-Specific Cartridges with Dual-View JSD Self-Distillation（工作稿）

[**论文 PDF**](main.pdf) · [LaTeX](main.tex) · [数字宏](macros.tex) · [主表](table_main.tex) · [统计](stats.json) · [作图脚本](make_figures.py)

Qwen3-4B-Instruct-2507（冻结），LongHealth / MTOB(KE) / QASPER，native KV 预算为 Whole KV 的 5/10/15/20%。
所有 Cartridge 方法使用相同的初始化、种子、15 条 trajectory 和训练配置（lr 0.04、accum 4、最多 32 次更新、相同的 plateau 停止规则），按 cell 逐一配对比较。
每个配对比较覆盖 15 个 request、57 个 cell。

## 完整结果表（逐 cell 匹配，n = 每个预算下所有核心方法都完成的 case 数）

Evidence ICL 和 Whole KV 不受预算影响。† 表示 MTOB 输出为乱码（baseline pipeline 问题），不是有效对比。FastKV 的 MTOB 20% 没有跑完。

**LongHealth Acc % ↑**

| KV compaction rate | n | Whole KV | Evidence ICL | SnapKV | FastKV (AM) | Vanilla Cart. | Vanilla+DualJSD | Specific+SD | Specific+DualJSD (Ours) |
|---|---|---|---|---|---|---|---|---|---|
| 5% | 15 | 86.7 | 80.0 | 86.7 | 60.0 | 33.3 | 40.0 | 53.3 | 60.0 |
| 10% | 14 | 85.7 | 78.6 | 85.7 | 78.6 | 64.3 | 57.1 | 64.3 | 64.3 |
| 15% | 14 | 85.7 | 78.6 | 85.7 | 85.7 | 50.0 | 50.0 | 71.4 | 71.4 |
| 20% | 14 | 85.7 | 78.6 | 85.7 | 85.7 | 42.9 | 71.4 | 57.1 | 71.4 |

**MTOB (KE) chrF ↑**

| KV compaction rate | n | Whole KV | Evidence ICL | SnapKV | FastKV (AM) | Vanilla Cart. | Vanilla+DualJSD | Specific+SD | Specific+DualJSD (Ours) |
|---|---|---|---|---|---|---|---|---|---|
| 5% | 15 | 6.0† | 27.3 | 5.8† | 4.1† | 13.9 | 18.6 | 19.7 | 17.5 |
| 10% | 14 | 6.0† | 27.7 | 5.7† | 4.1† | 17.0 | 17.3 | 20.0 | 17.1 |
| 15% | 14 | 6.0† | 27.7 | 5.8† | 6.4† | 17.4 | 17.0 | 17.5 | 18.9 |
| 20% | 14 | 6.0† | 27.7 | 5.8† | — | 17.0 | 13.8 | 19.2 | 15.0 |

**QASPER log-PPL ↓**

| KV compaction rate | n | Whole KV | Evidence ICL | SnapKV | FastKV (AM) | Vanilla Cart. | Vanilla+DualJSD | Specific+SD | Specific+DualJSD (Ours) |
|---|---|---|---|---|---|---|---|---|---|
| 5% | 15 | 2.77 | 3.76 | 2.82 | 2.75 | 2.02 | 1.90 | 1.83 | 1.64 |
| 10% | 14 | 2.80 | 3.86 | 2.76 | 2.63 | 1.93 | 1.90 | 1.78 | 1.57 |
| 15% | 14 | 2.80 | 3.86 | 2.81 | 2.68 | 2.09 | 1.84 | 1.88 | 1.29 |
| 20% | 14 | 2.80 | 3.86 | 2.77 | 2.60 | 1.97 | 1.89 | 1.62 | 1.30 |

## 一、为什么 query-specific 数据更好（相同 SD）

| | 胜/负/平 | 平均增益 |
|---|---|---|
| LongHealth acc | 13/5/39 | +14.0 分 |
| MTOB chrF | 35/21/1 | +2.8 |
| QASPER log-PPL | 40/17/0 | −0.22（越低越好） |

机制上，练习题覆盖 query 关键词的比例为 73% vs 16%（LongHealth）、47% vs 1%（MTOB），对抗了稀释问题。
reranker 把 gold 证据排在第一的比例为 62%（lexical 31%，random 10%），因此 Teacher context 含 gold 证据的比例是 45% vs 26%。

## 二、DualJSD 如何起作用、在哪里起作用（相同数据，与 SD 对比）

| | specific 数据 | agnostic 数据 |
|---|---|---|
| QASPER | **43/14/0**，log-PPL −0.32 | 35/22/0，−0.12 |
| LongHealth | 8/5/44，+5.3 分 | 6/2/49，+7.0 分 |
| MTOB | **24/28/5，chrF −2.0（变差）** | 29/27/1，+0.4（基本持平） |

Specific+DualJSD 对比 Vanilla Cartridges：LongHealth 17/6/34（+19.3 分），QASPER 47/10/0，MTOB 28/28/1（持平）。
在 QASPER 上，Specific+DualJSD 是所有预算下 log-PPL 最低的压缩方法（1.29–1.64，Whole KV 约 2.8，但其中一部分来自答案格式的适配）。
MTOB 上 W 对 E 的答案更没把握（答案平均 log-prob 为 −0.87 vs −0.16），把 W 混进 target 会稀释翻译需要的局部语法细节，所以 DualJSD 在这里变差。

注意：MTOB 的 Whole KV 和 SnapKV 输出是乱码（baseline pipeline 的问题），不能作为有效对比；本文没有做显著性检验；FastKV 的 MTOB 输出同样是乱码。
