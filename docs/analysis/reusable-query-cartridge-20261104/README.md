# 可直接调用的 Query Cartridge

`MaskedTTT.prepare_cartridge(request)` 沿用原有的选块、全文 Student、选中 Teacher 和 on-policy SD，返回独立卡带和第一次答题报告。卡带持有物化后的选中 KV、原问题后缀和冻结模型的引用；全文 KV 在首次回答前释放，选中源缓存与训练状态不再占用额外存储。未修改的选中 KV 层可以复用原有存储。后续 `cartridge.answer()` 用新的后缀缓存再次回答**同一个问题**，`cartridge.close()` 释放该请求的卡带引用。它不提供跨问题迁移，也不把 Teacher 的答案放进提示。

Qwen3 和 GPT-OSS 分组查询注意力的小模型测试覆盖了零更新和两步更新、重复及并发调用、KV 与权重不变、错误配置和释放。相关路径的 **60 项 CPU 测试通过**。离线构建的 wheel 也在两种模型架构、分数选块与关联选块下完成训练、独立 KV 物化和重复解码；这些检查不使用真实模型权重，不能代表 BABILONG 涨点。

真实模型的合成请求验证已接入现有 Qwen 和 GPT-OSS 作业的可选 GPU 预检，不增加新作业，也不修改其评测训练流程。它会比较原运行时的首次生成、卡带首次生成和重复生成的 token 与 logits，并检查模型配置、随机状态及缓存释放。只有作业到达该预检时才会执行；目前这里不宣称已通过真实 GPU 检查。

`document_kv_bytes` 统计物化后的文档 KV，不包括模型权重、问题后缀、生成后缀或训练峰值。完整的实际内存对照见[已有测量](../aligned-cartridge-memory-20261104/README.md)。

测试时发现 home 的文件数配额已满，DeepSpeed 在退出时无法创建 Triton 缓存锁文件。新生成的 GPU 作业脚本将 Triton 与 Torch 扩展缓存写入各作业的 scratch 目录；六个现有 Qwen／GPT 评测作业也安装了仅修改这两个缓存路径的启动配置。该配置对**之后启动的 Python 进程**生效，不改变已在运行的进程、模型或训练代码。为恢复 Git 同步，清理了本项目 57 个可自动重建的 Python 字节码文件，没有删除源代码或模型。[缓存路由代码](../../../experiments/gqa_head_gain/cache_environment.py)。

此外，原 home 下的 230 个 Triton 编译缓存文件已复制到 scratch，逐文件直接比较后迁移；原路径保留为指向新位置的符号链接。这样沿用默认路径的程序仍可访问缓存，并释放 home 的文件数配额。没有计算哈希，也没有删除模型。

[调用示例](../../../README.md#optional-linked-selection-and-kv-only-decoding) · [卡带实现](../../../src/masked_ttt/cartridge.py) · [预检代码](../../../experiments/kv_only_cartridge/reusable_probe.py) · [测试](../../../tests/test_reusable_cartridge.py)
