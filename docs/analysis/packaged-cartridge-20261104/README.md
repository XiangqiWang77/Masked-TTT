# Qwen 与 GPT-OSS 的离线安装包验证

已在独立临时目录、无网络条件下构建当前公开代码的 wheel，核对 CLI 入口，并从 wheel 而非源码目录运行 CLI 帮助和卡带方法。使用小型随机初始化模型与合成 Decision 分数，结果如下：

| 模型架构 | 分数选块 | 关联选块 | 本地未量化检查点加载 |
|---|---|---|---|
| Qwen3 | 通过 | 通过 | 通过 |
| GPT-OSS 分组查询注意力 | 通过 | 通过 | 通过 |

两种选块路径均检查了预期块、同一原文的缓存分数、Teacher→Student KL、两组可训练状态的梯度、模型与原始 KV 不变，以及全文 KV、选中源 KV 和训练状态在独立卡带解码前的释放。检查没有调用 Decision API，也没有加载真实模型权重。

这验证的是安装包和小模型端到端路径；真实模型 GPU 的数值和释放检查见[请求级预检](../request-local-gpu-preflight-20261104/README.md)，BABILONG 涨点由正在进行的独立评测决定。[离线构建脚本](../../../tests/run_wheel_smoke.py) · [小模型测试](../../../tests/wheel_smoke.py)。
