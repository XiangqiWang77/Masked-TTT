# DualJSD 为什么、怎样比 vanilla SD 好（Qwen3-4B，11/12 配对实验）

![DualJSD insights](fig_dualjsd_insights.png)

[PDF](fig_dualjsd_insights.pdf) · [SVG](fig_dualjsd_insights.svg) · [统计](insights_stats.json) · [作图](make_fig.py) · 数据：[token 级](tokens.py)、[cell 级](collect.py)、[时间](collect_time.py)

数据来自同一批请求的配对实验：DualJSD 与 SD 使用相同的轨迹、初始化、种子和训练配置，唯一的区别是训练 target。
共 342 个配对 cell（3 个数据集 × 2 种数据来源 × 4 个预算 × 每格 14–15 个请求），以及 87.8 万个答案 token。

**(a) DualJSD 只在证据视角 E 没把握的地方改 target。**
E 的 top-1 概率 > 0.95 的 token 约占三分之二，这些 token 的权重保持约 0.95，top-1 几乎不变（0.2–0.3%）。
E 的 top-1 概率 < 0.5 时，权重降到约 0.75，40–46% 的 token 被全文视角 W 改了 top-1。
所以 DualJSD 不是均匀地把 target 变软，而是一个由 E 的不确定性控制的纠错。

**(b) 在 LongHealth 和 QASPER 上事实 token 被保留，在 MTOB 上被稀释。**
数字 token 中权重 < 0.7 的只有 2–3%，DualJSD 基本不碰精确事实。
MTOB 上数字和专名（Kalamang 词、语法标注）有 30–34% 被大幅降权，因为全文视角对 E 的局部语法细节没把握（答案平均 log-prob 是 −0.87 对 −0.16）。
这正好对应 MTOB specific 上 DualJSD 变差。

**(c) 配对结果（win − loss 比例）。**
- QASPER specific 在 4 个预算上都是正的（+0.20 到 +0.71）。
- LongHealth 大多打平或小幅为正。
- MTOB specific 在 20% 预算上 −0.43。

每个请求的收益和 E/W 分歧程度之间只有弱正相关（specific：ρ = 0.17–0.44，n = 15，不显著）。因此这里只说"改在哪里、改了什么"，不说"分歧越大收益越大"。

**(d) 时间成本**（同一种 GPU；DualJSD 只改 target，每次更新的计算量和 SD 一样，实测中位数都是约 1.1 秒）：

| | SD 训练 / cell | DualJSD 训练 / cell | 两视角打分 / 请求（分摊到 4 个预算） | 更新次数 SD → DualJSD |
|---|---|---|---|---|
| LongHealth | 18.6 s | 17.7 s | 6.9 s（1.7 s / cell） | 30.6 → 29.0 |
| QASPER | 12.8 s | 12.8 s | 4.8 s（1.2 s / cell） | 30.3 → 30.3 |
| MTOB | 79.4 s | 64.9 s | 304 s（76 s / cell），其中 254 s 是 26 万 token 的全文 prefill | 28.1 → 22.9 |

DualJSD 的 target 更平滑，所以更早触发 training-loss plateau 早停：MTOB specific 96% 的 cell 提前停止，SD 大多跑满 32 步。
在 LongHealth 和 QASPER 上，总耗时和 SD 基本相同（每个 cell 多 1–2 秒）。只有 MTOB 这种超长文档，一次性的全文打分成为主要开销，每个 cell 多约 60 秒。

**注意事项**
- SD 和 DualJSD 原本在不同作业里运行（SD 一半跑在 B200 上），所以训练时间用"更新次数 × 同一 GPU 上实测的每步时间"来比较。
- 每格只有 14–15 个请求，没有做显著性检验。
- 权重 w 由存储的 w·M（保留至少 0.99 的概率质量）恢复，误差小于 1%。
