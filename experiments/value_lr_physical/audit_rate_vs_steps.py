#!/usr/bin/env python3
"""Compare two completed original-SD candidates on strictly matched cases."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.value_lr_physical.audit_value_strength import validate_pair


def summarize(rows):
    n = len(rows)
    def rate(key):
        return 100 * sum(row[key] for row in rows) / n if n else None
    def pair(before, after):
        good = sum(not row[before] and row[after] for row in rows)
        bad = sum(row[before] and not row[after] for row in rows)
        return dict(new_correct=good, new_wrong=bad,
                    delta_pp=100 * (good-bad) / n if n else None)
    return dict(n=n, accuracy={name: rate(name) for name in ("teacher", "zero", "rate", "step8")},
                rate_vs_teacher=pair("teacher", "rate"), step8_vs_teacher=pair("teacher", "step8"),
                rate_vs_zero=pair("zero", "rate"), step8_vs_zero=pair("zero", "step8"),
                rate_vs_step8=pair("step8", "rate"))


def audit(workspace):
    workspace = Path(workspace)
    step_root = workspace / "query_cartridges_step_dev_20261104"
    rate_root = workspace / "query_cartridges_value_lr_20261104"
    step_report = json.loads((step_root / "latest.json").read_text())
    rate_report = json.loads((rate_root / "latest.json").read_text())
    step_manifest = json.loads((step_root / "manifest.json").read_text())
    profile = json.loads((rate_root / "olmo_profile.json").read_text())
    if (step_report["metric"] != "answer_focused_lexical_v2" or rate_report["metric"] != step_report["metric"] or
        step_report["expected_per_model"] != 200 or rate_report["expected"] != 200 or
        not step_report["development_only"] or not rate_report["development_only"] or
        not step_report["evaluation_answers_used_only_posthoc"] or rate_report["evaluation_answers_used_in_training"] or
        step_report["errors"] or step_report["invalid"] or rate_report["errors"] or rate_report["invalid"] or
        step_report["online_api_calls"] != 0 or rate_report["online_api_calls"] != 0 or
        step_manifest["step_budgets"] != [2, 4, 8] or step_manifest["keep_fraction"] != .1 or
        step_manifest["selected_transfer_weight"] != 0 or step_manifest["reference_answers_used_in_inference"] is not False or
        profile["value_learning_rate"] != .0001 or profile["student_rollout_tokens"] != 8):
        raise ValueError("Both complete development reports must use the same scorer")
    models = {}
    for model in ("qwen", "gpt"):
        step_rows = step_report["models"][model]["rows"]
        rate_rows = rate_report["models"][model]["rows"]
        steps = {row["case"]: row for row in step_rows}
        rates = {row["case"]: row for row in rate_rows}
        if len(steps) != 200 or len(rates) != 200 or set(steps) != set(rates):
            raise ValueError("The development cases differ or are incomplete")
        matches, rows = Counter(), []
        for case in sorted(steps):
            x = json.loads((step_root / "cases" / model / case / "result.json").read_text())
            y = json.loads((rate_root / "cases" / model / case / "lr_0.0010.json").read_text())
            z = json.loads((rate_root / "cases" / model / case / "lr_0.0001.json").read_text())
            middle = json.loads((rate_root / "cases" / model / case / "lr_0.0003.json").read_text())
            validate_pair([z, middle, y], case)
            if (any(item["key"] != case or item["training"]["objective"] != "on-policy KL(teacher || student)" or
                    item["training"]["selected_transfer_weight"] != 0 or
                    item["training"]["student_rollout_limit"] != 8 or
                    not item["training"]["student_rollouts_refreshed_each_step"] or
                    not item["training"]["teacher_conditioned_on_student_prefix"] or
                    item["selection"]["online_api_calls"] != 0 or item["selection"]["reference_answers_used"] or
                    item["self_study"]["reference_answers_used"] or item["self_study"]["student_answer_hint_used"] or
                    item["teacher_answer_in_final_prompt"] or
                    not item["backbone_unchanged"] or not item["original_document_KV_unchanged"]
                    for item in (x, y, z)) or x["model"] != y["model"] or x["model"] != z["model"] or
                x["task"] != y["task"] or y["value_learning_rate"] != .001 or z["value_learning_rate"] != .0001):
                raise ValueError("The original-SD protocol changed")
            xp = x["standalone_cartridge"]["paired_prompt_control"]
            zp = z["standalone_cartridge"]["paired_prompt_control"]
            check = dict(selection=x["selection"]["selected_original_chunks"] == y["selection"]["selected_original_chunks"],
                teacher_tokens=x["teacher"]["tokens"] == y["teacher"]["tokens"],
                practice=x["self_study"]["checks"] == y["self_study"]["checks"],
                initial_prefix=x["training"]["steps"][0]["student_rollout_tokens"] == y["training"]["steps"][0]["student_rollout_tokens"],
                zero_tokens=xp["zero_original"]["tokens"] == zp["zero_original"]["tokens"],
                step2_tokens=x["standalone_cartridge"]["step_probe"]["checkpoints"]["2"]["tokens"] == zp["trained_original"]["tokens"])
            for key, valid in check.items():
                matches[key] += int(valid)
            if not all(check.values()):
                continue
            a, b = steps[case], rates[case]
            if a["teacher"] != b["teacher"] or a["zero"] != b["zero"]:
                raise ValueError("Matched answers have inconsistent scores")
            item = dict(teacher=a["teacher"], zero=a["zero"], rate=b["lr_0.0010"], step8=a["step8"])
            if any(type(value) is not int or value not in (0, 1) for value in item.values()):
                raise ValueError("The shared scorer must be binary")
            rows.append(item)
        models[model] = dict(matched_fields=dict(matches), strictly_matched=summarize(rows))
    return dict(metric=step_report["metric"], development_cases_per_model=200, source_questions=50,
        context_lengths_per_source=4, rate_value_learning_rate=.001, step_value_learning_rate=.0001,
        step_base_updates=8, scorer_used_post_generation_only=True,
        different_gpu_allocations_possible=True, new_model_calls=0, online_api_calls=0, models=models)


def render(data):
    lines = ["# Value 学习率与步数：同题配对", "",
        "这是已完成的 200 题开发集事后分析：每种模型 50 个来源问题，各有四种上下文长度。"
        "两种候选都使用原 Decision 分数前约 10% 的块和原式全文 Student 自蒸馏。", "",
        "| 模型 | 严格匹配题数 | Teacher | 选中零更新 | Value LR 0.001 | 8 步 | 0.001 相对 8 步：新对 / 新错 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("qwen", "gpt"):
        row=data["models"][model]["strictly_matched"];a=row["accuracy"];p=row["rate_vs_step8"]
        fmt=lambda value:"—" if value is None else f"{value:.2f}%"
        lines.append(f"| {model} | {row['n']} | {fmt(a['teacher'])} | {fmt(a['zero'])} | {fmt(a['rate'])} | "
                     f"{fmt(a['step8'])} | {p['new_correct']} / {p['new_wrong']} |")
    qn=data["models"]["qwen"]["strictly_matched"]["n"]
    gp=data["models"]["gpt"]["strictly_matched"]
    change=gp["rate_vs_step8"]
    lines += ["", "严格匹配要求同题的选块、Teacher token、已验证练习、初始学生前缀、"
        "零更新 token 和原学习率的第 2 步输出相同。两轮可能在不同 GPU 上运行；"
        f"GPT 有 {gp['n']} 题满足所有条件，Qwen 为 {qn} 题。训练和回答均未读取评测答案。"
        "Teacher 带通用回答要求，两个卡带候选使用原问题格式。开发集的配对输出来自选中 KV，"
        "但部分诊断状态当时仍在内存中；物理释放由留出实验另行验证。", "",
        f"在这批严格匹配的 GPT 题中，0.001 相对 8 步新答对 {change['new_correct']} 题、"
        f"新答错 {change['new_wrong']} 题；"
        "这支持先完成已提交的更强 Value 学习率验证。两种分别训练的结果并不能预测"
        "把高学习率与 8 步组合后的准确率。", "",
        "[聚合数据](summary.json)不含原文、答案或逐题 ID；本分析没有重新调用模型或 API。", ""]
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
