# Qwen 和 GPT-OSS 的真实模型卡带检查

这是 2026-11-04 10:48 UTC 的检查快照。检查使用真实模型和一条合成请求；不调用 Decision API，也不使用评测答案。它们检验卡带计算与释放，不衡量答题准确率。

| 模型 | GPU | 物化前后输出 | 32 位置 FlashAttention 梯度 | 源缓存与训练状态 |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | H200 | token 和 logits 一致 | 一致 | 已释放 |
| GPT-OSS-20B | B200 | token 和 logits 一致 | 一致 | 已释放 |

两项检查各比较了 3 个生成位置；物化卡带没有保留重复的适配参数。另一次 Qwen B200 检查比较了公开请求级入口与冻结实验路径，确认两步训练的前缀和梯度、最终 token 和 logits 一致；通用回答格式在该合成请求释放缓存前后也一致。

GPT-OSS 上述 B200 结果验证的是实验使用的 KV-only 物化路径。公开请求级入口在 GPT-OSS 上的独立 GPU 检查仍由现有作业执行，不能把 Qwen 的结果直接当作 GPT 的证明。[聚合数据](summary.json)来自已保存的预检结果，[脚本](../../../experiments/kv_only_cartridge/freeze_real_evidence.py)不运行模型。
