# 物化后只保留选中 KV

原独立 Cartridge 已将注意力偏置与 Value 修正写入选中缓存，但解码器仍持有一份用于构建该缓存的读取参数副本。物化完成后，增强注意力直接读取修改后的 K/V，不需要再次使用这份副本。

[可安装的实现](../src/masked_ttt/kv_only.py)只在独立卡带构造完毕、最终解码开始之前释放这份重复的参数和索引张量，保留修改后的选中 KV 以及轻量层路由信息。Python 接口可直接调用 `method.answer(request, compact_standalone=True)`，这一选项属于当前请求，不修改进程级解码钩子；它要求全文缓存和选中原始缓存均在独立解码前释放。该操作不改变选块、Teacher、SD、训练更新或物化得到的 K/V。

小模型配对测试覆盖 Qwen3 和 GPT-OSS 的原生零更新及非零适配路径，逐次检查预测分布和生成 token 是否一致，并核对释放后的字节数。[测试](../tests/test_kv_only_cartridge.py)也覆盖异常时恢复请求钩子。

历史 BABILONG 对照使用的物化与冻结训练流程保持不变；本请求级入口是相同精简操作的新调用方式。[真实模型预检](../experiments/kv_only_cartridge/real_model_probe.py)可在已经加载的 GPU 模型上验证同一个人工构造请求中各生成位置的 logits、答案及存储释放；新的请求级入口仍需在真实 GPU 模型上单独验证。

[可选预检驱动](../experiments/kv_only_cartridge/driver.py)支持复用现有单 GPU 作业中已经加载的模型，不再提交新 GPU 作业。除生成一致性以外，预检还在目标设备上检查 32 个读者位置的因果注意力与两类梯度。驱动按模型加锁并保存结果，结束时恢复随机数状态和原配置；预检失败不会替换正式预测。只有预检通过后，才考虑将这项精简用于正式卡带。

[配对评测路径](../experiments/kv_only_cartridge/paired_control.py)先生成同缓存的零更新与训练后对照，再释放这些对照的临时引用，调用实际的独立卡带生成。正式答案必须与训练后对照逐 token 一致，并验证全文、选中源缓存及训练状态均已释放。此前关联选块的开发集采用的是临时配对路径；它的准确率可以用于选块对照，但不应单独作为“配对评测时训练状态已释放”的证据。后续关联选块验证使用新的独立生成路径。

[原问题格式的配对路径](../experiments/kv_only_cartridge/original_controls.py)也可与关联选块及 32-token 学生前缀组合。针对 Qwen 和 GPT-OSS 的小模型端到端[检查](../tests/test_linked_long_rollout_physical.py)确认：两组读取参数均收到梯度，正式答案与同缓存训练后对照逐 token 相同，解码前全文、选中源缓存及训练参数被释放。这验证的是组合路径，不是两种真实模型在 BABILONG 上的准确率。
