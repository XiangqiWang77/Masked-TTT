# 两个真实纠错案例：JSD 选择性自蒸馏改变了什么？

这两例来自已保存的 BABILong 配对实验：**合成题保持不变，只比较 Vanilla SD 与 JSD-SD**。第一例显示，JSD-SD 保住了普通 SD 最后丢失的正确回答；第二例显示，JSD-SD 答对了保存的全文 Teacher 和普通 SD 都答错的问题。它们说明这两次选择性训练有实际答题收益，但不能据此宣称门控能够判定每条 Teacher 轨迹的真假。

这是 2026-11-11 历史实验的两条观察，使用 Qwen3-4B-Instruct-2507、512 个 native KV 槽位、学习率 0.02。它们不是后来新增的 Teacher 绝对置信度门控结果，也不是正在运行的 FastKV 或三数据集实验。

## 这两个例子里，算法到底做了什么？

两个方法读取同一套 query-specific 合成题、绑定证据和缓存 Teacher 软概率，使用同一个初始 Cartridge。基础模型冻结，训练只更新 Cartridge 的 K、V。

普通 SD 在全部练习轨迹上持续训练。JSD-SD 先执行完全相同的前半程训练，然后在这个共同的热身状态上，比较每道练习题的 Teacher 与 Student 预测分布。门控只改变后半程使用的轨迹集合；蒸馏损失和总更新次数保持不变。

对第 (i) 条回答轨迹：

$$
J_i=\frac{1}{|y_i|}\sum_t
\operatorname{JSD}_2(p_T^{i,t},p_S^{i,t})
\frac{H_2(p_S^{i,t})-H_2(p_T^{i,t})}
{H_2(p_S^{i,t})+H_2(p_T^{i,t})+10^{-12}},
\qquad m_i=\mathbf 1[J_i>0].
$$

这里的分布按实验实现对齐为 **Teacher top-k 词元加一个 rest 合并桶**。(J_i>0) 的轨迹进入后半程；其余轨迹不再用于后半程巩固。Teacher 相对更确定时倾向于放行，Student 相对更确定时倾向于屏蔽；这个方向是置信度信号，不是真假标签。

表格里的熵是整条轨迹的平均值，帮助阅读；真正的 (J_i) 是先逐 token 计算乘积再平均，所以不能用两个平均熵之差代替它。

## Example 1：保住 Fred，避免普通 SD 最终产生乱码

**请求：** `babilong-32-qa5-300105`，原实验标为 **development**。

**真实问题：** “Who did Mary give the apple to?”

**参考答案：** `Fred`。参考答案仅用于离线评分，没有进入门控。

绑定材料确实包含以下原句，已经在保存的证据文本中逐字核对：

> Mary picked up the apple there.
> Mary gave the apple to Fred.

### 生成了什么练习题？

本次训练有六道相关练习题，保存的合成回答全部是 `Fred`。但是，**相同的答案文本，在不同问题措辞下，Teacher 的软概率分布可以很不一样**。算法筛选的正是这些不同的软监督轨迹。

下表列出全部六条记录；索引从零开始。

| 索引 | 合成问题 | 合成回答 | Teacher 平均熵 | Student 平均熵 | ED-JSD 分数 | 后半程 |
|---:|---|---|---:|---:|---:|---|
| 0 | Who received the apple from Mary? | Fred | 0.701112 | 0.468857 | −0.069451 | 屏蔽 |
| 1 | To whom did Mary pass the apple she picked up? | Fred | 0.001324 | 0.034834 | +0.001549 | 保留 |
| 2 | What person did Mary give the apple to after picking it up? | Fred | 0.083107 | 0.334289 | +0.010585 | 保留 |
| 3 | Who was the recipient of the apple that Mary gave? | Fred | 0.208556 | 0.649699 | +0.074122 | 保留 |
| 4 | Who received the apple after Mary picked it up? | Fred | 0.069438 | 0.504562 | +0.126743 | 保留 |
| 5 | To whom did Mary pass the apple she found? | Fred | 0.001234 | 0.053971 | +0.002693 | 保留 |

### 怎么训练，最后发生了什么？

两组都训练八步，前四步的训练损失逐步完全一致。JSD-SD 在第四步后屏蔽第 0 条轨迹，随后用其余五条执行剩余四步；Vanilla SD 继续使用全部六条。

| 读取方式或训练结果 | 保存的最终输出 | 评分 |
|---|---|---|
| 初始 Cartridge | `Fred` | 正确 |
| 保存的全文 Teacher | `Fred` | 正确 |
| Vanilla SD | 乱码输出，完整文本见下方 | 错误 |
| JSD-SD | `Fred` | 正确 |

Vanilla SD 的原始输出如下，按原样保存，没有替换成某个推测答案：

```text
, or
,, T
,,
,, Tel
 Tel Tel
,,
,  1 1 Fo Fo T T Tint
Fo, 1 0
```

**这例支持什么？** 在相同数据和更新预算下，选择性后半程训练保住了初始 Cartridge 已经具备的正确回答，而普通 SD 最终产生了无效输出。这与减少相对不可靠软监督的干扰相符。

**不能把它解释成“删掉了一道错误答案”。** 被屏蔽的第 0 条，其答案也是正确的 `Fred`；它被屏蔽是因为 ED-JSD 分数为负。这例更准确的意义是：答案文本正确，并不意味着它对应的每一种 Teacher 软分布都适合继续灌入 Student。

这里观察到的是完整选择性训练方案的结果，没有做单条删除、位置保持或重打包的独立消融。因此，无法仅凭日志把乱码消失归因于某一个 token 或某一种数值机制。

## Example 2：保存的全文 Teacher 和 SD 都答 no，JSD-SD 答 yes

**请求：** `babilong-64-qa6-300147`，原实验标为 **holdout**。

**真实问题：** “Is John in the garden?”

**参考答案：** `yes`，仅用于离线评分。

某些绑定证据确实含有原句：

> John went back to the garden.

但另一些候选材料混入了叙事背景中的 `John Fleming`，或只出现一句 `Allee same, John`。这些材料生成的练习题和回答并不都能有效支持真实问题，因而这例的学习集合存在冲突与无关内容。

### Teacher 提供了什么监督，JSD 如何处理？

这次共有二十四条轨迹。下面列出能说明不同门控行为的代表记录；全部二十四条都保存在配套 JSON 中，没有隐藏其他保留或屏蔽决定。

| 索引 | 合成问题 | 保存的合成回答 | Teacher 平均熵 | Student 平均熵 | ED-JSD 分数 | 后半程 |
|---:|---|---|---:|---:|---:|---|
| 0 | Did John return to the garden according to the evidence? | Yes, John is in the garden. | 0.743264 | 0.658528 | −0.000839 | 屏蔽 |
| 1 | Is there a statement confirming John's presence in the garden? | Yes, John is in the garden. | 0.423929 | 0.605848 | +0.004034 | 保留 |
| 7 | Can you identify a sentence that confirms John's presence in the garden? | Yes, John is in the garden. | 0.498689 | 0.545776 | +0.000285 | 保留 |
| 8 | Is there any mention of John being present in the garden according to the provided text? | No, John is not in the garden. | 0.453910 | 0.439790 | +0.000592 | 保留 |
| 12 | Did John go to the garden after being in the kitchen? | Yes, John is in the garden. | 0.420808 | 0.469596 | +0.001524 | 保留 |
| 16 | Is John Fleming associated with the garden based on the given information? | John is mentioned as John Fleming, but there is no information confirming whether he is in the garden. | 0.631186 | 0.633223 | −0.000031 | 屏蔽 |
| 18 | Can we determine from the evidence whether John is present in the garden? | John is mentioned as John Fleming, but there is no information confirming whether he is in the garden. | 0.565691 | 0.706538 | +0.004497 | 保留 |

这一组记录很重要：门控确实屏蔽了部分无关的 John Fleming 轨迹，但同类轨迹也有被保留的；支持 yes 的练习也有被屏蔽的。**不能把这次纠错写成“JSD 把错误答案全部删掉，只留下正确答案”。**

按保存的回答文本做一个简单分类，可以看清实际变化。以下是文本类别计数，不是逐题真实性标注；问题本身不同，不能直接对 yes/no 做多数投票。

| 合成回答文本类别 | 全部轨迹 | 后半程保留 | 后半程屏蔽 |
|---|---:|---:|---:|
| `Yes, John is in the garden.` | 12 | 4 | 8 |
| `No, John is not in the garden.` | 4 | 4 | 0 |
| 提到 John Fleming，但称无法确定其位置 | 4 | 2 | 2 |
| 针对某一句背景引文作答 `No` | 4 | 4 | 0 |
| 合计 | 24 | 14 | 10 |

### 怎么训练，最后发生了什么？

两组都执行二十六步更新，前十三步的训练损失逐步完全一致。之后 Vanilla SD 继续训练全部轨迹；JSD-SD 使用十四条保留轨迹完成后十三步，没有触发空集合回退。

| 读取方式或训练结果 | 保存的输出 | 离线评分 |
|---|---|---|
| 初始 Cartridge | `$no$` | 错误 |
| 保存的全文 Teacher | `no` | 错误 |
| Vanilla SD | `$no$` | 错误 |
| JSD-SD | `$yes$` | 正确 |

**这例支持什么？** 对这个请求，JSD-SD 的最终结果超过了保存的全文 Teacher 输出，也超过了相同合成数据上的普通 SD。Teacher 对原始问题的一次错误预测，没有成为最终 Cartridge 的表现上限。

**可以怎样解释？** 学习发生在多个问题条件下的软概率分布上；JSD 改变了后半程哪些分布继续贡献更新，最终改变了原始问题的回答。它不是对训练回答的 yes/no 次数做投票，因此保留池里仍有 no 或信息不足的轨迹，并不与最终生成 yes 矛盾。

**目前不能确定什么？** 没有逐条删除消融，也没有保存每条轨迹对最终 yes/no 概率的独立因果贡献。因此，只能说整个门控训练方案在这例产生了纠错，不能确认是哪一条被删轨迹导致纠错，更不能声称所有被保留的轨迹都是正确监督。

## 对 “Beyond Oracle Trap” 应该怎样表述？

两个例子应分别使用：

- **Example 1：减少训练退化的观察。** 原始 Teacher 和初始 Cartridge 都答对，但普通 SD 最后出错，JSD-SD 保住正确答案。
- **Example 2：超过保存的全文 Teacher 的观察。** 全文 Teacher、初始 Cartridge 和普通 SD 都答错，JSD-SD 最后答对。

适合论文的表述是：

> On these paired cases, ED-JSD selective self-distillation changes which teacher-conditioned trajectories are reinforced, preserving a correct answer in one case and recovering the correct answer despite an incorrect saved full-context teacher prediction in another. The gate measures relative uncertainty and disagreement, not factual correctness.

这两例是已完成样本中的案例分析，不是独立的效果保证。第一例属于开发样本，第二例的原始标记为 holdout；全文 Teacher 的输出由该实验从以前记录复用，并非为本次案例报告重新生成。

## 数据与核对方式

- [两例的完整原始输出、门控记录、训练历史和原文引用定位](correction_cases.json)
- [全部配对请求及结果](paired_case_results.csv)
- [全部被屏蔽练习题](blocked_trajectories.csv)
- [配对 target hash、更新数、热身损失核查](paired_audit.json)
- [分布图、配对结果图与完整说明](README.md)

两例的 Teacher target SHA256 在各自 SD/JSD-SD 配对之间一致，热身阶段逐步损失完全相同。原文短引句已在保存的绑定材料中验证，JSON 记录其字符位置和源文件 SHA256。参考答案只用于离线评分，没有添加进合成题、门控或训练。

## 写作核查

| 主张 | 证据 | 范围 |
|---|---|---|
| 选择性训练在两例中改善了最终答案 | 相同输入条件下的保存输出及原始评分 | 支持这两例 |
| 第一例属于保住已有正确能力 | 初始输出 Fred；SD 乱码；JSD-SD Fred | 支持 |
| 第二例超过了保存的全文 Teacher | Teacher no；SD no；JSD-SD yes；参考 yes | 支持这例，Teacher 输出为复用记录 |
| 被屏蔽轨迹都是错误答案 | Fred 和多条 yes 轨迹也被屏蔽 | 不支持 |
| 单独删除某一条轨迹造成了纠错 | 没有对应独立消融 | 尚不能确认 |

审阅结论：贡献以实际配对行为表述；保留反例与历史协议信息；开发/holdout 标签不混淆；不从两例外推完整测试集；将相对确定性、答案正确性和单条轨迹的因果作用分开说明。
