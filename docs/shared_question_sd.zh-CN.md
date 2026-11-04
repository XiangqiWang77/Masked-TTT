# 同问题条件下的 Query Cartridge 自蒸馏校准

## 发现

原有实验中的 ICL Teacher 在原问题后额外收到一条通用输出格式要求：

> Return only the final answer in the required format. Do not add an explanation.

全文 Student 的训练后缀没有这条要求。因此，Teacher 与 Student 的分布差异同时包含证据视图和输出格式的差异。该现象本身不证明它是表现下降的唯一原因，需要同题实验验证。

## 校准

`experiments/shared_question_sd/alignment.py` 将原有的通用格式要求移动到共享的问题提示中，Teacher 不再单独追加同一句话。这样保留原来的 Teacher 输入、模型、Decision API 缓存分数和选中 chunks，同时使全文 Student、Teacher 与最终 Cartridge 的问题后缀一致。Teacher 仍独立读取选中原文；Student 仍在完整文档 KV 上进行 on-policy KL 自蒸馏，只更新选中位置的注意力偏置和值读取修正。原模型和原始缓存冻结；没有增加全文到选中缓存的反向损失，也没有在选中缓存上训练。

`experiments/shared_question_sd/controls.py` 在完全相同的选中 KV、问题后缀和生成预算下保存零更新和更新后的诊断答案。正式答案仍走原本的独立 Cartridge 物化路径：先释放全文 KV，然后物化选中缓存，释放选中源缓存及训练适配器，最后解码。答案标签仅在生成完成后用于评测，Teacher 答案不放入最终问题。

## 验证与实验

- Qwen3 与 GPT-OSS 的小模型测试检查 Teacher 输出不变、两侧训练后缀一致、两个参数组和原缓存的约束，以及正式解码前的释放状态。
- 使用两个模型的真实 tokenizer 和冻结的 Decision 分数，对 8K、16K、32K、64K 示例验证选块与 Teacher 输入的构造。
- 用 BABILONG 每模型 200 道开发题进行同题比较：全文、选中零更新、ICL Teacher、校准后的 Cartridge，以及原来以相同格式生成的 Cartridge。再根据开发结果选择是否在按来源问题隔离的 800 道题上验证。

这是一项有针对性的校准实验；是否提高最终回答准确率，由配对结果决定。
