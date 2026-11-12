# Contextual Cartridges

当前完整方法已同步到 [ACL 方法稿](paper/contextual_cartridges/METHOD.zh-CN.md)。

当前版本是 Answer-Guided Query-Specific Dataset Crafting 与 ED-JSD Selective SD：半预算预热后对每条候选轨迹做一次前向，在 (k+1) 桶粗化分布上计算熵方向 Jensen–Shannon 分数，严格大于零才进入后半程巩固；结构性 Source-Support Gate（空回答、截断、显式拒答）在打分前排除退化轨迹。无候选通过时精确回退为普通 SD。保留原生 KV、冻结 backbone 和稀疏软目标 SD；无 held-out 标签，选择过程零反向传播，不保证最终答案零退化。
