# Contextual Cartridges

当前完整方法已同步到 [ACL 方法稿](paper/contextual_cartridges/METHOD.zh-CN.md)。

当前版本是 Answer-Guided Query-Specific Dataset Crafting 与 Conservative TracIn-Guided Selective SD：三个检查点一致负向提名，再用两种固定顺序实际删样本重训确认。保留原生 KV、冻结 backbone 和稀疏软目标 SD；无 held-out 标签。守护对象为保留轨迹的 SD 损失，不保证最终答案零退化。
