#!/usr/bin/env python3
"""Audit matched 1,000-case results by task, without running a model or API."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(rows):
    if not rows:
        raise ValueError("A task group may not be empty")
    names = ("teacher", "selective", "cartridge", "selective_teacher_format", "cartridge_teacher_format")
    if any(type(row[name]) is not int or row[name] not in (0, 1) for row in rows for name in names):
        raise ValueError("Expected binary paired scores")
    n = len(rows)
    counts = {name: sum(row[name] for row in rows) for name in names}
    def changes(before, after):
        return dict(new_correct=sum(not row[before] and row[after] for row in rows),
                    new_wrong=sum(row[before] and not row[after] for row in rows))
    return dict(n=n, counts=counts,
                original_update=changes("selective", "cartridge"),
                teacher_format_update=changes("selective_teacher_format", "cartridge_teacher_format"))


def freeze(olmo, qwen, gpt):
    if (olmo["model"] != "OLMo-3-7B" or olmo["dataset"] != "BABILONG" or
        olmo["expected"] != 1000 or olmo["overall"]["n"] != 1000 or olmo["errors"] or
        olmo["reference_answers_used_only_in_reporting"] is not True):
        raise ValueError("A complete matched OLMo rerun is required")
    expected = set(olmo["case_ids"])
    if len(expected) != 1000:
        raise ValueError("The OLMo case identities are incomplete")
    lengths_by_source = defaultdict(set)
    for case in expected:
        fields = case.split("-", 2)
        if len(fields) != 3 or fields[0] != "babilong" or fields[1] not in ("8", "16", "32", "64"):
            raise ValueError("Unexpected BABILONG case identity")
        lengths_by_source[fields[2]].add(fields[1])
    if len(lengths_by_source) != 250 or any(lengths != {"8", "16", "32", "64"} for lengths in lengths_by_source.values()):
        raise ValueError("The four context lengths must share 250 source questions")
    models = {}
    for name, report in (("qwen", qwen), ("gpt", gpt)):
        if (report["metric"] != "answer_focused_lexical_v2" or report["completed"] != 1000 or
            report["errors"] or report["reference_answers_used_only_posthoc"] is not True or
            len(report["rows"]) != 1000 or {r["case"] for r in report["rows"]} != expected):
            raise ValueError("Complete same-case paired reports with the common scorer are required")
        groups = defaultdict(list)
        for row in report["rows"]:
            if row["task"] != row["case"].split("-", 2)[2].split("-", 1)[0]:
                raise ValueError("Case and task identities differ")
            groups[row["task"]].append(row)
        models[name] = dict(overall=summarize(report["rows"]),
                            by_task={task: summarize(rows) for task, rows in groups.items()})
    tasks = set(olmo["by_task"])
    if any(set(model["by_task"]) != tasks for model in models.values()):
        raise ValueError("Task identities differ across the models")
    for task in tasks:
        if any(model["by_task"][task]["n"] != olmo["by_task"][task]["n"] for model in models.values()):
            raise ValueError("Task counts differ across the models")
    models["olmo"] = dict(overall=olmo["overall"], by_task=olmo["by_task"])
    return dict(metric="answer_focused_lexical_v2", source_questions=250,
        context_lengths_per_question=4, new_model_calls=0, online_api_calls=0,
        teacher_has_generic_output_requirement=True, original_cartridge_format=True,
        olmo_zero_control="independent rerun with same selected KV and original suffix",
        qwen_gpt_zero_control="same selected KV and original suffix",
        teacher_format_control="same selected KV, generic output requirement and generation limit",
        models=models)


def render(data):
    models = data["models"]
    def original(model, task):
        x = model["by_task"][task]
        c = x["counts"]
        return " → ".join(f"{c[name] / x['n'] * 100:.0f}" for name in ("teacher", "selective", "cartridge"))
    def formatted(model, task):
        x = model["by_task"][task]; c = x["counts"]
        return " → ".join(f"{c[name] / x['n'] * 100:.0f}" for name in ("selective_teacher_format", "cartridge_teacher_format"))
    lines = ["# 按题型检查 OLMo 涨点与跨模型复现", "",
        "每个模型使用相同的 1,000 道 BABILONG 题目，统一使用宽松词面评分。"
        "以下每种题型 100 道，来自 25 个来源问题的四种上下文长度；表中数字为准确率（%）。", "",
        "表中 `Teacher → 零更新 → TTT` 依次表示选中片段 Teacher、独立选中缓存的零更新、"
        "同一缓存与原问题格式下的训练后输出。Teacher 额外有通用回答要求；"
        "因此前后两段分别描述方法级差异和同格式更新差异。", "",
        "| 题型 | OLMo：Teacher → 零更新 → TTT | Qwen：Teacher → 零更新 → TTT | GPT-OSS：Teacher → 零更新 → TTT |",
        "|---|---:|---:|---:|"]
    for task in sorted(models["olmo"]["by_task"], key=lambda task: int(task[2:])):
        x = models["olmo"]["by_task"][task]; c = x["correct"]
        olmo = " → ".join(f"{c[name] / x['n'] * 100:.0f}" for name in ("teacher", "zero", "trained"))
        lines.append(f"| {task.upper()} | {olmo} | {original(models['qwen'], task)} | {original(models['gpt'], task)} |")
    lines += ["", "下面进一步保持同一缓存、同一 Teacher 通用输出格式和生成长度，"
              "比较 Qwen 与 GPT-OSS 的零更新及训练后输出。", "",
        "| 题型 | Qwen：零更新 → TTT | GPT-OSS：零更新 → TTT |", "|---|---:|---:|"]
    for task in sorted(models["olmo"]["by_task"], key=lambda task: int(task[2:])):
        lines.append(f"| {task.upper()} | {formatted(models['qwen'], task)} | {formatted(models['gpt'], task)} |")
    lines += ["", "OLMo 表使用独立配对复跑，整体 Teacher 41.00%、零更新 44.30%、TTT 44.50%；"
              "最初那轮分别为 41.20% 和 44.60%。两轮的 Teacher 都有额外格式要求。"
              "在本次复跑的 QA4 与 QA10 中，零更新相对 Teacher 分别高 14 和 10 题，"
              "TTT 相对相同格式的零更新分别再高 2 和 0 题。",
              "", "GPT-OSS 的 QA10 原问题格式为 38% → 40%；改用通用格式时为 50% → 51%。"
              "格式会改变模型回答内容，这一事后对比用于设计后续固定设置，不能当作新的留出集结果。",
              "", "没有新的模型或 API 调用。参考答案只用于已生成回答的统一评分；"
              "[聚合数据](summary.json)不包含原文、答案或逐题身份。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def read(name):
        return json.loads((args.workspace / name / "latest.json").read_text())
    data = freeze(read("query_cartridges_zero_control_20261103"),
                  read("query_cartridges_qwen_olmo1000_20261104"),
                  read("query_cartridges_gpt_olmo1000_20261103"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "README.md").write_text(render(data))


if __name__ == "__main__":
    main()
