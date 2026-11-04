# 选中证据的 token 映射检查

使用 Qwen3-4B 和 GPT-OSS-20B 各 40 道 BABILONG 开发题，分别核对已有 Decision JSON 对应的选块。审计只加载本地分词器，不加载模型、不调用 API、不读取参考答案；40 题来自 10 个来源问题的四种长度。

| 模型 | 选中 token 与原文逐一对应 | sink 归属改变 | 与重新分词后的 token 序列完全相同 | 原序列 token 数 → 重新分词 token 数 |
|---|---:|---:|---:|---:|
| Qwen3-4B | 40/40 | 0/40 | 1/40 | 120,195 → 119,897 |
| GPT-OSS-20B | 40/40 | 0/40 | 1/40 | 118,168 → 117,872 |

当前实现保留选中 chunk 的原 token ID，在不连续片段之间插入分隔符，再用这份序列独立 prefill Teacher 和卡带。审计中，每个选中 token 都正确映射回全文原位置；拼接文本经重新分词、解码后也保持相同文字，没有产生替换字符。全文提示的 token 数与已保存的模型运行记录在两种模型上均为 40/40 一致。

文本一致并不意味着 token 序列唯一：当前拼接方式与对拼接文字重新分词的序列在 39/40 题不同。表中最后一列是总长度变化，**不是 token 错误率**；不同序列也可能长度相同。这是一个值得做配对实验的工程差异，现有数据不能说明它会提高或降低答题准确率。

如果后续测试重新分词，需要同时重新建立“全文 chunk → 选中 token”的映射，特别处理分隔符与证据共用一个 token 的边界，保证训练得到的修正仍施加到正确片段。本审计没有改动正在运行的方法。[聚合数据](summary.json) · [复核脚本](../../../experiments/value_lr_physical/audit_selected_tokenization.py) · [映射测试](../../../tests/test_selected_tokenization_audit.py)。

进一步按重新分词后的字符 offset 检查，两个模型各 **40/40** 题的选中 chunk 边界均可稳定恢复，没有 token 同时跨两个选中 chunk，也没有非空白证据字符漏覆盖。Qwen 有 299 个、GPT-OSS 有 297 个 token 同时覆盖证据和片段外的边界文字，这些 token 分布在各自的 39 道题中。它们需要在候选实现中明确规定归属；不能把它们当作完全独立的分隔符。

这项可行性检查只对选中证据片段重新分词，没有验证把整个聊天提示连同前缀和问题一起重新分词时的边界，也没有运行模型。对于字符边界落在不完整 UTF-8 字节序列中的情况，脚本会明确标记为不支持，而不是猜测归属；这批题没有出现该情况。

已据此实现一个**独立、默认关闭**的候选：[canonical_evidence.py](../../../experiments/value_lr_physical/canonical_evidence.py) 重新分词证据片段，并把只跨越“分隔文字＋一个选中 chunk”的 token 归属到该 chunk；跨两个选中 chunk、无法恢复字符边界或涉及 sink 的情况直接拒绝。两种模型的各 40 题都能构建候选计划。

[canonical_context.py](../../../experiments/value_lr_physical/canonical_context.py) 让候选 Teacher 和最终卡带使用相同的新证据序列，全文 Student 的原 SD 和更新参数不变。Qwen 与 GPT-OSS 的小模型测试覆盖了无变化时的一致性、实际边界合并、参数冻结和缓存释放。该候选没有进入当前作业，真实模型准确率尚未测试；它也没有对整个聊天提示重新分词。

扩展的 CPU 可行性检查覆盖每个模型固定开发集的 **200 题**：Qwen 和 GPT-OSS 均为 **200/200** 可构造，且各有 197 题的证据 token 序列与原拼接方式不同。两者分别有 1,458 和 1,451 个 token 跨证据与分隔文字边界，已依照上述规则归属。检查核对了各题与已完成运行的全文提示长度和选中原文块，没有加载模型、读取参考答案或调用 API。这仅验证候选的输入构造，不是准确率评测。[聚合数据](development_feasibility.json) · [复核脚本](../../../experiments/value_lr_physical/canonical_development_audit.py)。
