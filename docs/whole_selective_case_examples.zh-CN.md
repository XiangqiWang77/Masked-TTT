# 全文与选中缓存：三道真实案例

下面的案例来自 Qwen3-4B 的 BABILONG 配对实验，选块复用保存的 Decision API 分数。全文、选中缓存和训练后的独立 Cartridge 使用同一道题；选中缓存的训练前后使用相同原问题后缀和生成上限。它们展示需要解决的问题和当前方法的实际行为，不把单题输出当成对内部因果机制的证明。

## 全文视野有时会漏用可见证据

题目 `babilong-16-qa1-300023` 问：“Where is John?” 文档中有 “John went back to the garden.”，这句话位于第 43 个块，已被选中。全文模型回答无法确定 John 的位置；零更新的选中缓存和训练后 Cartridge 都回答 garden。

这道题说明，全文里存在证据并不保证回答时一定用到；更聚焦的视角在此题更有利。该日志没有记录这次回答的注意力分布，因此不能单凭答案断言哪个注意力头发生了弥散。

## 片段都选到了，关系仍可能没被正确使用

题目 `babilong-16-qa8-300185` 问：“What is John carrying?” 选中块中按原文顺序包含两条事实：

1. 第 23 块：“John took the football there.”
2. 第 26 块：“John put down the football.”

问题需要根据先拿起、后放下的关系回答 nothing。全文模型回答 nothing；选中 Teacher、零更新的选中缓存和训练后 Cartridge 均回答 football。

这不是简单的“没有选中事实”：两条事实都进入了独立选中缓存，但选中视角下的回答没有正确反映后一次状态变化。不能仅凭输出断言是切块、注意力还是推理中的哪一环导致错误；可以直接观察到的是，现有 Cartridge 在此题没有把全文路径的正确结果恢复出来。

## 连接事实被漏选时，视角会出现真正的信息缺口

题目 `babilong-32-qa2-300037` 问：“Where is the milk?” 相关原文的时间顺序如下：

1. 第 63 块：“Daniel grabbed the milk there.”——选中。
2. 第 114 块：“Daniel moved to the garden.”——未选中。
3. 第 121 块：“Daniel dropped the milk.”——选中。

全文回答 garden；选中 Teacher、零更新的选中缓存和训练后 Cartridge 均回答 bathroom。只保留首尾两条时，决定最后落点的移动事件不在独立选中缓存的输入中。这道题展示了选择召回对独立 Cartridge 的影响；当前这次更新没有恢复正确答案。

## 与 Masked TTT Cartridges 的关系

方法在构造时同时使用两个视角：Teacher 阅读重新拼接的选中片段；Student 阅读全文，并在选中位置学习注意力与 Value 读取修正。完成后，修正被映射到独立 prefill 的选中 KV，全文 KV 在独立回答前释放。优化目标是选中 Teacher 对全文 Student 的 on-policy 蒸馏，而不是直接要求选中缓存复现全文模型的分布。因此，全文参与训练为信息迁移创造条件，但现有目标不自动保证把缺失关系或事实恢复到 Cartridge。

配对汇总见 [KV Whole 与 KV Selective 的互补性](dual_view_complementarity_snapshot.zh-CN.md)。案例中的选块、题目及输出来自对应运行日志；标准答案只用于生成后的评估。
