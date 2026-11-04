# 32-token 学生前缀：开发对照

2026-11-04 04:21 UTC；BABILONG 统一宽松词面评分。

| 模型 | 题数 | 全文 | 选中零更新 | ICL Teacher | Cartridge | 新答对 / 新答错（相对零更新） |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 38 | 50.00 | 42.11 | 44.74 | 47.37 | 2 / 0 |
| gpt | 0 | — | — | — | — | 0 / 0 |

| 模型 | 严格配对题数 | 8-token SD | 32-token SD | 新答对 / 新答错 |
|---|---:|---:|---:|---:|
| qwen | 36 | 50.00 | 50.00 | 0 / 0 |
| gpt | 0 | — | — | 0 / 0 |

Teacher 和 Student 保留原 OLMo 方法的输出提示；Teacher 看到独立选中原文，Student 在全文 KV 上按原 SD 更新，最终仅使用物化的选中缓存。
严格配对要求 Teacher token、选中块、零更新 token、自学习候选及学生原始前缀一致；不一致的结果不会作为严格配对。
答案标签仅在生成后用于评分；缓存比例不包括模型、构造峰值或问题后缀。
qwen 严格配对 36 题，其中 5 题的学生前缀更长。
qwen 开发选择：development_or_matched_baseline_incomplete。
gpt 严格配对 0 题，其中 0 题的学生前缀更长。
gpt 开发选择：development_or_matched_baseline_incomplete。
