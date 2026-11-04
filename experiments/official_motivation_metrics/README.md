# OLMo：论文支持的两项动机指标

这份单 GPU 诊断复用原 OLMo Query Cartridge 运行时、冻结的 Decision 分数与 BABILONG 原始支持事实标注。脚本不调用在线 API，不修改原 Teacher 设置；原问题、全文适配、选中 KV 物化与最终回答保持历史方法。金标准支持事实和参考答案只用于**训练结束后的测量**。

## 两个指标

1. **Target attention mass（支持事实注意力质量）**。使用 [qTTT 论文 Appendix E](https://arxiv.org/html/2512.13898) 的定义：对答案位置，将标注目标 token 的注意力概率求和，再平均所有层、头和答案位置。这里的目标集合推广为 BABILONG 所有标注支持事实的 token。比较全文、未更新的选中 KV 和 QC-TTT。图的左轴报告该值，越高表示支持事实得到的注意力越多；它是 score dilution 的直接读取指标，不是答案正确率。
2. **Complete-evidence recall（完整证据召回率）**。按 [Retrieval from Within / INTRA](https://arxiv.org/html/2605.05806) 对 complete-evidence recall 的定义，统计全部标注支持证据都被保留的问题比例；本实验将其应用于 BABILONG 的支持事实字符区间，而非该论文的 oracle chunk 标签。支持事实 recall 的标准定义也可见 [HotpotQA 官方评测代码](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)。图的右轴报告该值，越低说明选块遗漏完整支持事实的比例越高。

完整证据召回衡量文本证据是否保留，**不直接测事实关系是否断裂或记忆中是否编码了缺失信息**。ICL KV 与 QC-TTT 使用同一批选中文本，因此此指标在两者上相同；不能用作区分两者内部读取能力的指标。

## 配对和样本

- 准确证据覆盖取 OLMo 保留完整工作文档、支持区间无歧义的 **838** 道请求：8K／16K／32K 分别 250，64K 为 88。其余 162 个 64K 工作文档事先压缩，未纳入覆盖率分母。
- 注意力测量为固定随机种子、按任务分层选择的 **12 个来源问题 × 4 种长度 = 48 次请求**；三个方法同题、同原问题后缀，使用同一参考答案前缀进行事后 teacher forcing，以保证对应位置可比。不会按最终准确率选择样本。
- ICL KV 是和 QC 使用相同选中缓存、相同原问题后缀的零更新读取，作为测量控制；它不是此前带额外格式指令的历史 Teacher 输出。QC 仍使用原 Teacher 进行训练。
- OLMo 使用滑动窗口与全文混合注意力。诊断逐层遵守原模型的可见范围，并用部署时的 Q/K 和已物化的注意力偏置以 FP32 重建最后一个 query 的 softmax；逐头求目标质量后按层与答案位置平均。QC 探针会检查生成的首 token 与不插桩时一致。
- 保存每例数据，注意力仅在一个批次完整结束时写入；若作业时间结束前未完成，图名和图内都标为 partial。图中小点为单例，大点为均值；右轴完整召回使用 838 题。三个面板的左右轴尺度各自固定，不连接缺失值。

## 代码与输出

`prepare.py` 用原始标注和固定种子建立样本清单；`attention_mass.py` 通过只读 forward pre-hook 计算目标质量；`run.py` 在原本自蒸馏与生成之后执行测量；`plot.py` 输出独立的三联 PNG／PDF／SVG。`tests/test_official_motivation_metrics.py` 检查 softmax、滑窗、物化偏置、GQA 与小型 OLMo 读出的等价性。

单个作业和生成的图、`figure_data.csv`、`figure_status.json` 位于：

`/nfs/roberts/scratch/pi_mg269/da839/stack_ttt/official_motivation_metrics_20261104/`

该方案由 Scientific Visualization skill 辅助设计；见 [Kassis 等（2026）](https://arxiv.org/abs/2609.00065)。
