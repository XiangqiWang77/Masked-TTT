# Query Cartridges：OLMo 提升版本的 In-Context Self-Distillation

## 这版方法是什么

用问题相关的原文片段构建一个 ICL Teacher，让它在 Student 自己的回答前缀上给出下一词分布；Student 读取完整工作文档，通过自蒸馏调整选中片段的 KV 读取状态。训练后，用独立 prefill 的选中片段 KV 加上这份状态形成 Query Cartridge，并独立回答原问题。

**Teacher 是训练信号，不把它的完整答案直接交给 Student。** 基础模型权重和原始文档 KV 都被冻结，所有适配状态只属于当前请求。

## 为什么这样设计

全文包含更多背景，但当前问题需要的信息可能分散在很多无关内容中。选中的片段为同一个模型提供更聚焦的阅读视角；自蒸馏把该视角下的预测分布变成 Student 的内部读取更新。

Teacher 给出的不只是一个最终词，还包括其他候选词的概率。Student 在当前生成前缀上对齐这份分布，因此每一步可以使用 Teacher 对多个候选的相对偏好。Teacher 也可能选错或漏掉证据，所以最终效果需要在同一批问题上验证。

## 实际执行流程

1. **选择证据。** Decision API 对文档块打分；本次 OLMo 配置复用已保存的分数，按分数选择约 10% 的块，再按原文顺序排列。
2. **建立两个阅读视角。** 冻结模型对完整工作文档做 prefill 得到全文 KV；同一模型另行对选中片段做 prefill，作为 ICL Teacher。Teacher 接收原问题与只输出最终答案的格式指令。
3. **构造训练问题。** 使用原问题；此外，Teacher 最多提出两个小事实问题，要求引用选中原文，并用另一次只看引用的回答核验。小题的候选答案只用于核验，原始 OLMo 版本没有把它作为额外答案提示交给 Teacher 或 Student。
4. **联合更新读取状态。** Student 使用全文 KV 生成短前缀；Teacher 使用选中片段和同一前缀给出下一词分布。用 Teacher 到 Student 的全词表前向 KL，联合更新选中位置的注意力分配和 Value 读出修正。所有问题共享同一份状态，模型权重不更新。
5. **生成独立回答。** 清除训练后缀。用独立构建的选中片段 KV 和训练好的读取状态重新处理原问题，生成 Query Cartridge 的答案；它在回答时不读取全文 KV。

完整工作文档还会产生另一条回答：全文 KV 加适配状态。原始配置用全文 Judge 比较这条回答与未适配回答；**这个 Judge 的选择不替代独立 Cartridge 的答案**。本报告中 44.60% 指独立 Cartridge。

## 自蒸馏的计算

记完整工作文档为 \(D\)，选中证据为 \(E\)，问题为 \(Q\)，请求内读取状态为 \(\phi\)。Student 每步从当前状态生成前缀 \(\tilde y_{<i}\)，然后冻结 Teacher 在同一前缀上给出分布：

\[
\mathcal L_{\mathrm{SD}}(\phi)=
\mathbb E_i D_{\mathrm{KL}}\!\left(
p_{\theta}(\cdot\mid E,Q,\tilde y_{<i})
\;\middle\|\;
p_{\theta,\phi}(\cdot\mid D,Q,\tilde y_{<i})
\right).
\]

小练习也采用相同结构：两侧使用同一个小问题和 Student 前缀，但 Teacher 读选中证据、Student 读完整工作文档。KL 的梯度同时流向两类读取状态。实际 Teacher 的主问题后缀还包含回答格式指令；比较结果时需要保留这一差异。

在选中位置，读取分数增加块级偏置，Value 读出增加按块低秩修正。未选中的文档位置仍可在全文 Student 中正常读取；适配只作用于选中位置。实现对全注意力层进行适配，滑动注意力层保持原生行为。

## OLMo 的 1,000 题结果

采用同一套任务感知 loose 词面判分、同一批 BABILONG 题：

| 方法 | 准确率 |
|---|---:|
| Vanilla | 36.10% |
| qTTT 式对照 | 36.00% |
| 同选块 ICL Teacher | 41.20% |
| 独立 Query Cartridge | **44.60%** |

独立 Cartridge 相对 Teacher 高 **3.40 个百分点**，逐题为新答对 58 题、新答错 24 题。其中 162 题因 OLMo 上下文窗口限制使用了压缩后的工作文档；Teacher 与 Student 使用对应的同一份工作文档。

这一结果比较的是完整方法配置。另一次 1,000 题的同选中缓存配对复跑中，零更新为 44.30%、更新后为 44.50%，更新本身相差 **0.20 个百分点**。Teacher 的回答格式指令和生成上限与独立 Student 不同，因此完整的 3.40 点不能直接全部归因于梯度更新。

## 复用这个配置

此分支基于产生 OLMo 结果的归档实现；核心训练、Teacher 和缓存路径保持原样。新增的配置文件记录了当时的 10% 选块、两步基础更新、rank 8、学习率、窗口和 Teacher 设置。

~~~bash
masked-ttt --model /path/to/Olmo-3-7B-Instruct \
  --request request.json --score-json-dir /path/to/shared_scores \
  --case-id CASE_ID --config-json examples/olmo_babilong_icl_sd.json \
  --local-files-only --output result.json
~~~

输出中的 prediction 是全文回答，standalone_cartridge.prediction 是独立选中 KV 的回答；两者不能混成同一条结果。训练阶段使用全文 KV，因此独立回答时的缓存压缩并不等于整个请求的峰值显存或总耗时降低。
