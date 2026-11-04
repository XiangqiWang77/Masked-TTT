#!/usr/bin/env python3
"""Group existing paired answers by first-step KL, without training or API calls."""
import argparse
import json
import math
from pathlib import Path

from experiments.value_lr_physical.audit_practice_signal import load_model


def summarize(rows):
    n = len(rows)
    def pair(before, after):
        gains = sum(not row[before] and row[after] for row in rows)
        losses = sum(row[before] and not row[after] for row in rows)
        return dict(new_correct=gains, new_wrong=losses,
                    delta_pp=100 * (gains-losses) / n if n else None)
    return dict(n=n, original_tokens_changed=sum(row["original_tokens_changed"] for row in rows),
        teacher_format_tokens_changed=sum(row["teacher_format_tokens_changed"] for row in rows),
        original_update=pair("selective", "cartridge"),
        teacher_format_update=pair("selective_teacher_format", "cartridge_teacher_format"))


def audit(workspace):
    result = dict(metric="answer_focused_lexical_v2", kl_threshold=.01,
        grouping="first main-question SD step; descriptive only; not a training gate",
        online_api_calls=0, new_model_calls=0, reference_answers_used_only_posthoc=True, models={})
    for model in ("qwen", "gpt"):
        rows = load_model(workspace, model)
        if len(rows) != 1000 or any(not math.isfinite(row["first_kl"]) for row in rows):
            raise ValueError("Expected 1,000 complete cases with finite first-step KL")
        low = [row for row in rows if row["first_kl"] < .01]
        high = [row for row in rows if row["first_kl"] >= .01]
        result["models"][model] = dict(overall=summarize(rows), low=summarize(low), high=summarize(high))
    return result


def render(data):
    lines = ["# 首步 Teacher KL 与卡带回答变化", "",
        "读取已完成的每模型 1,000 道 BABILONG 日志；每 1,000 题由 250 个来源问题扩为四种长度。"
        "0.01 是描述性分组阈值，不是训练门控；每题训练仍使用原有全文 Student 和选中 Teacher 的 KL。", "",
        "| 模型 | 首步 KL | 题数 | 原格式 token 序列变化 | 原格式新对 / 新错 | 同 Teacher 格式新对 / 新错 |",
        "|---|---:|---:|---:|---:|---:|"]
    for model in ("qwen", "gpt"):
        for key, label in (("low", "< 0.01"), ("high", "≥ 0.01")):
            item=data["models"][model][key]; a=item["original_update"]; b=item["teacher_format_update"]
            lines.append(f"| {model} | {label} | {item['n']} | {item['original_tokens_changed']} | "
                         f"{a['new_correct']} / {a['new_wrong']} | {b['new_correct']} / {b['new_wrong']} |")
    lines += ["", "每列的新对与新错都比较同一选中缓存、相同问题格式和生成长度的零更新与训练后答案。"
        "首步 KL 衡量选中 Teacher 与全文 Student 在其前缀上的分布差异，不是正确性分数。"
        "不同题型和提示会影响 KL；该分组用于定位训练信号，不能直接确定最佳更新参数。", "",
        "Qwen 的低 KL 组几乎没有生成变化，说明值得检验更长的学生前缀是否让答案位置获得训练信号；"
        "当前更长前缀和提示对齐的配对实验负责验证最终准确率。", "",
        "[聚合数据](summary.json)不包含原文、答案或逐题身份。没有新模型或在线 API 调用。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=audit(args.workspace)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(data,indent=2)+"\n")
    (args.output/"README.md").write_text(render(data))


if __name__=="__main__":
    main()
