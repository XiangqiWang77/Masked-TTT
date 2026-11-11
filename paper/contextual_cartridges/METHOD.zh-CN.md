# Contextual Cartridges: Query-Guided Test-Time Learning with Selective Self-Distillation

Contextual Cartridges 针对已经到来的真实问题，把长文档中与该问题有关的知识写入请求内的原生 KV 记忆。它只有两项技术：**Answer-Guided Query-Specific Dataset Crafting** 决定学什么，**Conservative TracIn-Guided Selective Self-Distillation** 决定哪些蒸馏轨迹值得保留。基础模型始终冻结，训练沿用 Cartridges 的 KV 参数化和软目标蒸馏，最终只挂载训练后的 KV 回答问题。本文描述的是当前代码中的“多检查点一致负向提名＋实际删样本重训确认”版本，不是旧版“两检查点分数相加后直接过滤”。

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

**第二项技术：默认保留，只删除得到一致、实测负向证据的轨迹。** 设所有可用轨迹为 $R_0$。从相同初始化出发，用预先固定的两种训练顺序 $b\in\{1,2\}$ 分别训练普通 SD，得到 $Z_{R_0}^{(b)}$。主顺序的普通训练确定更新次数 $T$；后续所有候选重训保持同一初始化、同一学习率、相同 $T$ 和新建的 Adam 状态。BABILong 四臂实验由 query-agnostic 普通 SD 决定两套数据共享的 $T$。匹配的是最终返回模型的更新次数，不是选择过程的总计算量。

对当前保留集 $R$，在初始化及两种顺序训练后的三个检查点 $\mathcal C_R=\{Z_0,Z_R^{(1)},Z_R^{(2)}\}$ 计算每条轨迹的 KV 梯度，并用其余轨迹的等权平均作为参考：

$$g_{i,c}=\nabla_Z\mathcal L_i(Z_c),\qquad \bar g_{-i,c}=\frac1{|R|-1}\sum_{j\in R\setminus\{i\}}g_{j,c},\qquad u_{i,c}=\bar g_{-i,c}^{\top}g_{i,c}.$$

排除候选自身，避免它的梯度范数给自己制造正分。在小步 SGD 的一阶近似下，正内积表示学习 $i$ 有望降低其余轨迹的平均损失，负内积表示有望增加该损失。实际优化器是 Adam，因此该分数只是影响代理，并不精确等于实际多步收益，也不判定事实真假。本方法借鉴 TracIn 的检查点梯度思想，但不直接使用原论文的检查点加权和。

只有三个检查点都满足 $u_{i,c}<0$，轨迹 $i$ 才进入待删除名单；任何正分、零分、检查点分歧或参考不足，都默认保留。候选按跨检查点最不负的 cosine 从小到大排序，仅用于确定尝试顺序。仅剩一条轨迹时照常训练和保留。

随后真正尝试删除候选：令 $A=R\setminus\{i\}$，从同一个 $Z_0$、用两种固定顺序分别重新训练 $T$ 步。对每个保留轨迹 $j\in A$，两个重训结果都必须满足：

$$\mathcal L_j(Z_A^{(b)})\le\mathcal L_j(Z_R^{(b)})+\epsilon(\mathcal L_j(Z_R^{(b)})),$$
$$\mathcal L_j(Z_A^{(b)})\le\mathcal L_j(Z_{R_0}^{(b)})+\epsilon(\mathcal L_j(Z_{R_0}^{(b)})),$$

其中 $\epsilon(x)=10^{-6}+10^{-5}|x|$ 是数值容差。除此之外，每种顺序下，保留轨迹的平均损失相对当前拟合都必须改善超过平均容差。第一组条件防止当前保留轨迹受损，第二组条件防止连续删除积累容差漂移；任何一个保留轨迹或训练顺序不通过，就撤销这次删除。

确认删除后更新 $R$，重新计算参考梯度，再检查后续候选，直到没有删除通过。最终固定返回主训练顺序得到的 KV，不按 benchmark 答案挑选种子。如果一次删除都未通过，直接复用已经训练完的主顺序 Vanilla SD 检查点，照常回答；不弃答，也不停止更新。这就是当前“trade compute for accuracy”的具体实现：多花计算验证删除对已定义学习目标的影响，而不是把梯度符号当成保证。

**这怎样针对 Oracle trap，以及能保证什么。** 算法允许拒绝与其余监督持续冲突、且被实测删除确认的 Teacher 轨迹，因而不再无条件复制全部轨迹。如果 Teacher 在原问题上推理失败，但在多道相关练习上提供了可整合的知识，Student 有机会通过学习这些知识获得更好的回答。多条轨迹也可能共享同一错误，删除后最终问题的准确率仍可能下降。当前守护的是“保留轨迹的 SD 损失”，不是“真实答案正确性”；无法在没有真值的情况下保证所有原来答对的请求绝不改错。没有确认删除时的精确回退，只保证该分支与普通 SD 完全一致。

**如何验证。** 固定同一批请求，运行 query-agnostic/query-specific × Vanilla/Selective SD 四臂；同时报告改善和退化次数、准确率净差及全部选择开销。基准答案只由生成答案之后的独立评价程序读取。最终推理释放原文和训练数据，只使用 $p_{\theta,Z^*}(\cdot\mid q)$。三数据集 cache 曲线每个容量必须使用相同的评估请求集合；LongHealth 用 accuracy，MTOB (KE) 用 corpus chrF，QASPER 用平均答案 log-perplexity，不能混为同一种 accuracy。

当前完成的 26,451 道/方法、1,000 个请求的数据审计显示支持材料覆盖和共同曝光改善；它没有证明所有练习答案正确、必须跨块推理、端到端更快或最终准确率提高。BABILong 与三模型 cache 图在本论文包中均保留待填状态，不混入旧配置的结果。

参考：[Cartridges 原论文](https://arxiv.org/abs/2506.06266)、[官方代码](https://github.com/HazyResearch/cartridges)、[TracIn 原论文](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html)。当前选择器见 [conservative_tracin.py](code/implementation/conservative_tracin.py)，运行逻辑见 [conservative_tracin_runtime.py](code/implementation/conservative_tracin_runtime.py)。
