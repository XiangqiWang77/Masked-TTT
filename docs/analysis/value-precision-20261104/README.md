# KV 物化时的小幅修正

GPT-OSS 的固定开发快照中，181 道已完成的配对题使用了同一模型、Teacher、选中缓存和 SD 目标，分别训练三个 Value 学习率。原始学习率 0.0001 下，抽样 Value 元素中位数只有 23.85% 在 BF16 写入后发生改变；写入误差的平方范数中位数为预期修正平方范数的 62.87%。这给出了一个具体可测的数值问题：一些较小的训练修正，在最终选中 KV 物化时可能被舍入。

较大学习率的写入误差更小，但它们同时改变了优化过程，不能把准确率变化直接归因于精度。新的配对测试会将同一训练状态分别写成 BF16 和 FP32 选中 Value，检查质量、零更新对照和 Cartridge 大小；FP32 使用不同注意力后端，因此也要报告对应的 FP32 零更新结果。

[固定快照表](gpt_write_error.md) · [聚合数据](gpt_write_error.json) · [分析代码](../../../experiments/babilong_value_lr/audit_quantization.py) · [测试](../../../tests/test_value_quantization_audit.py)
