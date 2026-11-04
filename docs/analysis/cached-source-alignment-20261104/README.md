# 缓存选块与原始请求的核对

Qwen3-4B 和 GPT-OSS-20B 的当前 BABILONG 协议指向同一份原始题目索引、JSONL 和 Decision 分数目录。对 1,000 个题目逐一检查后，题目 ID、问题、分块位置与分数结构均可读取并通过校验；8K、16K、32K、64K 各 250 题。本次没有调用 API、读取评测答案或计算哈希。

新接口还允许调用者提供原始索引和 JSONL，在读取缓存分数前直接比较输入文档和问题。如果同一问题被配到了不同文档，调用会立即失败。该校验确认请求与提供的索引一致；缓存文件本身仍需来自该索引所对应的打分批次。

[汇总](summary.json) · [核对脚本](../../../experiments/value_lr_physical/audit_cached_source_alignment.py) · [调用方式](../../../README.md#run)
