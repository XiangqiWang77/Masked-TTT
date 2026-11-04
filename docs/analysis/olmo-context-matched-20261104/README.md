# OLMo 涨点与上下文覆盖的配对检查

三种模型使用相同的 1,000 个 BABILONG 题目 ID、各自已生成的回答和同一套宽松词面评分。下面按 **OLMo 的工作文档是否保留完整原文**分组：838 题完整，162 题由于 OLMo 窗口限制先压缩。该分组只描述 OLMo 的输入，不能假定另外两种模型的工作文档与其完全相同。

| OLMo 工作文档 | 题数 | OLMo：Teacher → Cartridge | Qwen：Teacher → Cartridge | GPT-OSS：Teacher → Cartridge |
|---|---:|---:|---:|---:|
| 完整 | 838 | 42.12% → **45.58%**（+3.46 点） | 58.11% → 57.64%（−0.48 点） | 58.47% → 56.56%（−1.91 点） |
| 先压缩 | 162 | 36.42% → **39.51%**（+3.09 点） | 48.77% → **51.85%**（+3.09 点） | 50.00% → 49.38%（−0.62 点） |

OLMo 在完整原文组仍有 3.46 点提升，所以它的总体涨点并非只来自那 162 道预先压缩的题。原始 Teacher 与 Cartridge 的通用输出格式要求不同，这个方法级比较不能把全部差值归因于训练。

在完整原文的 838 个相同题目上，Qwen 与 GPT-OSS 的**同格式选中缓存零更新 → 训练后**分别是 58.11% → 58.35% 和 58.47% → 58.71%。这说明跨模型复现需要检查训练状态怎样影响最终卡带，而不能仅复制 OLMo 的超参数。

[summary.json](summary.json) 包含各组配对新答对／新答错数量；[脚本](../../../experiments/value_lr_physical/audit_olmo_context_overlap.py) 在生成后读取已有结果并校验题目 ID，没有新模型或 API 调用。OLMo 源数据和完整覆盖定义见 [原始固定快照](https://github.com/XiangqiWang77/Masked-TTT/blob/experiment/cartridge-basis-alignment-20261104/docs/analysis/olmo-motivations-20261104/README.md)。
