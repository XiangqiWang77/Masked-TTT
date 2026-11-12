# Contextual Cartridges: Query-Guided Test-Time Learning with Selective Self-Distillation

Contextual Cartridges 针对已经到来的真实问题，把长文档中与该问题有关的知识写入请求内的原生 KV 记忆。它只有两项技术：**Answer-Guided Query-Specific Dataset Crafting** 决定学什么，**ED-JSD Selective Self-Distillation**（熵方向 Jensen–Shannon 门控）决定哪些蒸馏轨迹进入后半程巩固。基础模型始终冻结，训练沿用 Cartridges 的 KV 参数化和软目标蒸馏，最终只挂载训练后的 KV 回答问题。本文描述的是当前代码中的“预热后一次前向、粗化分布 ED-JSD 严格正阈值”版本，不是早期基于梯度影响分数的版本；旧版本不再出现在本方法包中。

**是什么、为什么。** 设输入文档为 $D=(C_1,\ldots,C_N)$，真实问题为 $q$，冻结模型参数为 $\theta$，可训练记忆为 $Z=(K_{\mathrm{cart}},V_{\mathrm{cart}})$。Dilution 指有限练习预算和 KV 容量被与当前问题关系弱的知识占用；fragmentation 指分别学习局部事实，并不保证练习覆盖当前问题需要的跨块关系。原版 Cartridges 本身已有结构性、全局推理提示，我们针对的是通用任务分布对一个已知问题的监督不足，而不是声称原版完全不能跨块推理。Oracle trap 则指无条件吸收所有 Teacher 轨迹可能把冲突或误导性的监督共同写入 KV；它不是“Student 准确率不能超过 Teacher”的理论上界。

**第一项技术：先挑材料、再提炼答案、再反推问题。** 给定 $(D,q)$，轻量 Decision 模型或已有 API 分数结合词面、实体和邻接/桥接召回，得到少量相关单块或互补块组 $E_i$。冻结生成器先从 $(q,E_i)$ 中提炼有助于回答 $q$ 的结论或中间答案 $a_i$，再据此反推闭卷练习题 $s_i$，保存问题、答案和原文位置：

$$P(E,a,s\mid D,q)=P(E\mid D,q)P_\theta(a\mid E,q)P_\theta(s\mid E,q,a),\qquad \mathcal S_q=\{(s_i,a_i,E_i)\}_{i=1}^{M}.$$

后两个语义步骤可以合并在一次生成请求中按顺序输出。材料无法提供有用结论时允许跳过；普通的空项、重复或明显泄露过滤负责数据格式，不额外运行昂贵的事实核验循环。围绕真实问题提炼结论，使练习预算集中于有关内容；由互补片段形成联合结论，再围绕结论反推问题，为跨块关系提供直接学习机会。但是模型提炼的结论仍可能错误，多块来源也不自动证明每道题都必须使用全部块。

当前 BABILong 实验直接使用已保存的两套 JSON，轻量路由来自缓存 Decision API 的单块分数和确定性的实体、桥接及邻接召回；块组不是 API 核验过的充分证据。三数据集 cache 运行器使用 Qwen3-Reranker-0.6B 召回/重排及其源块组。两者共享以答定题原则，检索适配器不同，不能把当前 BABILong 数据描述成全部重新由该 reranker 生成。

**沿用的训练基础：只更新原生 KV 的 SD。** 对每条轨迹，Teacher 看绑定材料和练习题，Student 看 KV 和同一道题，两者沿用相同回答前缀：

$$p_T^{i,t}=p_\theta(\cdot\mid E_i,s_i,y_{i,<t}),\qquad p_S^{i,t}=p_{\theta,Z}(\cdot\mid s_i,y_{i,<t}).$$

概念目标为 Teacher 到 Student 的 KL：

$$\mathcal L_i^{\mathrm{KL}}(Z)=\frac1{|y_i|}\sum_t D_{\mathrm{KL}}\!\left(\operatorname{sg}[p_T^{i,t}]\;\middle\|\;p_S^{i,t}\right).$$

代码沿用官方稀疏软目标损失：保留的 Teacher token–词表条目组成 $\Omega_i$，Student 在完整词表上计算 log-softmax，Teacher 的保留概率不重新归一化：

$$\mathcal L_i(Z)=-\frac1{|\Omega_i|}\sum_{(t,v)\in\Omega_i}\operatorname{sg}[p_T^{i,t}(v)]\log p_S^{i,t}(v;Z).$$

这是稀疏概率加权交叉熵近似，不能写成代码精确计算完整词表 KL。完整 Teacher 交叉熵与完整 KL 对 Student 的梯度相同，但 top-k 截断仍有近似。当前 BABILong 保存的 synthetic assistant answer 本身就是固定轨迹 $y_i$，Teacher 在这些前缀上提供软目标；所以不能说 $a_i$ 在出题后被完全丢弃，也不能把这种训练称为以 $a_i$ 为真值的硬标签 NLL。三数据集运行器则在绑定材料上生成 Teacher 轨迹，并跨 cache 容量复用。初始化 $Z_0$ 来自官方 KV 初始化，不是抽象的空白张量；其冻结前缀位置继续冻结。

**第二项技术：ED-JSD Selective Self-Distillation（前向只读门控，精确预算）。** 设预算为 $T$ 次普通 SD 更新，由对照臂的数据布局决定并在两臂严格相等；四个 cache 容量复用同一批轨迹。前 $h=\lfloor T/2\rfloor$ 步在全部轨迹 $R$ 上做普通 SD 预热。候选池先经过确定性的 Source-Support Gate：在 Teacher 生成时记录的结构检查（问题或绑定材料缺失、空回答、截断、显式拒答）未通过的轨迹不参与打分，但仍参与预热和全量回退；该门不判定语义真伪。在预热得到的固定 KV 处，对每条剩余候选只做一次前向（整个选择过程零反向传播），使用与 Teacher 完全相同的回答前缀。

每个回答位置保留 Teacher 的 top-$k$ 概率与对应的全词表归一化 Student 概率，再把两侧各自的剩余质量并成一个 rest bucket，得到 $(k{+}1)$ 桶粗化分布；缺失的 Teacher 概率不当作零，保留质量也不重新归一化。以 2 为底、取 $M=\tfrac12(T+S)$，逐 token 计算

$$J=\Big[H(M)-\tfrac12\big(H(T)+H(S)\big)\Big]\times\frac{H(S)-H(T)}{H(S)+H(T)+10^{-12}},$$

对轨迹取 token 均值，严格 $J>0$ 才进入巩固集 $A^+$。前一因子是有界的 JSD（分歧大小），后一因子给分歧一个方向：只有 Student 比 Teacher 更不确定时才为正。可选的披露性置信条件另外要求 Teacher 保留 top-1 概率的几何均值（不含 rest bucket）不低于给定阈值；这是置信度启发式，不是正确性判定。

其余 $T-h$ 步只在 $A^+$ 上继续，沿用同一个 sampler 与同一 Adam 状态。若 $A^+$ 为空，默认策略继续原数据、原优化器不变，在确定性前向下逐位等于跑满 $T$ 步的普通 SD；可选的 retain_warmup 策略改为保留预热 KV，并如实报告更少的实际更新数。若全部通过，同样保持原 sampler 不变。整个选择过程没有第二次拟合、没有梯度或跨轨迹影响比较、没有 held-out 参考损失、没有 benchmark 标签、没有硬标签答案损失、没有 RL；训练目标始终是上游稀疏软目标 SD，JSD 只是选择统计量。粗化分布的熵与 JSD 不是全词表统计量：合并尾部可能隐藏尾部内的分歧，也可能改变表观熵序。

**这怎样针对 Oracle trap，以及能保证什么。** 门控不再无条件复制全部 Teacher 轨迹：结构上退化的轨迹（截断、拒答、空回答）在打分前被排除，预热后只巩固“分歧明显且 Teacher 比 Student 更确定”的轨迹。但置信不是正确——自信而错误的 Teacher 可以通过此门，自信而错误的 Student 也可能让正确的 Teacher 信息被拒绝；多条轨迹也可能共享同一错误。守护对象是分布层面的选择统计量，不是真实答案正确性，无法在没有真值的情况下保证原来答对的请求绝不改错。$A^+$ 为空时的精确回退只保证该分支与普通 SD 完全一致。

**如何验证。** 固定同一批请求，运行 query-agnostic/query-specific × Vanilla/Selective SD 四臂；同时报告改善和退化次数、准确率净差及全部选择开销。基准答案只由生成答案之后的独立评价程序读取。最终推理释放原文和训练数据，只使用 $p_{\theta,Z^*}(\cdot\mid q)$。三数据集 cache 曲线每个容量必须使用相同的评估请求集合；LongHealth 用 accuracy，MTOB (KE) 用 corpus chrF，QASPER 用平均答案 log-perplexity，不能混为同一种 accuracy。

当前完成的 26,451 道/方法、1,000 个请求的数据审计显示支持材料覆盖和共同曝光改善；它没有证明所有练习答案正确、必须跨块推理、端到端更快或最终准确率提高。BABILong 与三模型 cache 图在本论文包中均保留待填状态，不混入旧配置的结果。

参考：[Cartridges 原论文](https://arxiv.org/abs/2506.06266)、[官方代码](https://github.com/HazyResearch/cartridges)。当前选择器见 [ed_jsd.py](code/implementation/ed_jsd.py)，结构源门见 [source_gate.py](code/implementation/source_gate.py)，英文技术说明见 [ED_JSD_SD.md](ED_JSD_SD.md)。
