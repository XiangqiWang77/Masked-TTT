# Qwen3-4B BABILong synthetic datasets

两套数据使用 Qwen3-4B-Instruct-2507。这里的“题”指一份 BABILong 文档与真实问题 `(D,q)`；每题会产生多道合成练习题。

| JSON | BABILong题数 | 合成训练题 | 合成验证题 | 合成题总数 | 覆盖状态 |
|---|---:|---:|---:|---:|---|
| [query_specific_syns.json](https://github.com/XiangqiWang77/Masked-TTT/releases/download/babilong-syn-qwen3-4b-1000-matched-32605413/query_specific_syns.json) | 1000 | 20558 | 7391 | 27949 | 1000/1000 完成 |
| [vanilla_cartridges_syns.json](https://github.com/XiangqiWang77/Masked-TTT/releases/download/babilong-syn-qwen3-4b-1000-matched-32605413/vanilla_cartridges_syns.json) | 1000 | 20558 | 7391 | 27949 | 1000/1000 完成 |

## 数据生成

`query_specific_syns.json`：根据真实问题挑选相关材料，先提取答案或中间结论，再反推练习题；包含24题已有数据和976题本轮补齐数据，保留各题的生成版本标记。

`vanilla_cartridges_syns.json`：已保存的原版 Cartridges 通用合成对话，出题不以真实问题为条件。当前保存1000题；`missing_benchmark_case_ids`列出缺失题号。每题的`query`仅用于关联评测，不表示原版生成时看到了该问题。

## JSON结构

每个case保存`query`、`task`、`length`、`train`、`validation`和`evidence_sets`。每条练习题包含`question`、模型合成的`answer`与`evidence_id`；在同一case的`evidence_sets`中查找该ID，即可取回原始证据文本。原版的多轮对话另存`conversation_messages`。

只对重复证据做存储去重，保留问答及证据内容；本机路径和调用日志未放入发布文件。答案是模型合成值，不包含benchmark隐藏gold答案。

两套JSON用于合成数据对照；vanilla synthetic不参与SD比较。SD实验只在固定的query-specific数据上比较Vanilla SD与Selective SD，匹配KV容量与训练预算。

生成基础：[Stanford Cartridges](https://github.com/HazyResearch/cartridges)。基准：[BABILong](https://github.com/booydar/babilong)。

完整JSON作为GitHub Release附件保存，避免大文件占用Git历史；[下载两套完整数据](https://github.com/XiangqiWang77/Masked-TTT/releases/tag/babilong-syn-qwen3-4b-1000-matched-32605413)。
