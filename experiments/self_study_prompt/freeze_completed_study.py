#!/usr/bin/env python3
"""Freeze aggregate results from the finished same-GPU Qwen study comparison."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.self_study_prompt.study_gate import development_decision, probe_decision
from experiments.babilong_linked_selection.long_rollout_protocol import validate_partition
from experiments.gqa_head_gain.report import summarize


def freeze(manifest, report, probe, preflight, finished, submission):
    validate_partition(manifest["case_ids"], manifest["holdout_case_ids"])
    rows = report["rows"]
    if (len(rows) != 200 or len({row["case"] for row in rows}) != 200 or
        {row["case"] for row in rows} != set(manifest["case_ids"]) or
        any(type(row[key]) is not int or row[key] not in (0, 1)
            for row in rows for key in ("whole", "zero", "teacher", "baseline", "cartridge"))):
        raise ValueError("Scores must cover the fixed 200-case development set once")
    groups = defaultdict(list)
    for row in rows:
        parts = row["case"].split("-", 2)
        if (len(parts) != 3 or parts[0] != "babilong" or parts[1] not in ("8", "16", "32", "64") or
            row["task"] != parts[2].split("-", 1)[0] or row["length"] != int(parts[1]) * 1024):
            raise ValueError("A scored task or context length does not match its case")
        groups["task:" + row["task"]].append(row)
        groups["length:" + str(row["length"])].append(row)
    computed, computed_groups = summarize(rows), {key: summarize(group) for key, group in groups.items()}
    if computed != report["overall"] or computed_groups != report["groups"]:
        raise ValueError("The saved answer aggregates differ from their scored rows")
    decision, probe_result = development_decision(report), probe_decision(probe)
    if (manifest["mode"] != "development" or manifest["models"] != ["qwen"] or
        manifest["case_count"] != 200 or len(set(manifest["case_ids"])) != 200 or
        manifest["selection_policy"] != "decision_entity_link_v1" or manifest["keep_fraction"] != .1 or
        manifest["self_study_prompt_comparison"] != "original_vs_study_only" or
        manifest["reference_answers_used_in_inference"] is not False or manifest["online_api_calls"] != 0 or
        report["reference_answers_used_only_posthoc"] is not True or report["online_api_calls"] != 0 or
        report["overall"]["n"] != 200 or report["overall"]["strictly_paired"]["n"] != 200 or
        report["errors"] or report["invalid"] or not probe_result["eligible"] or
        not preflight["passed"] or not preflight["paired_study_preflight"] or
        not preflight["same_gpu"] or not preflight["independently_materialized_and_released"] or
        not finished["all_complete"] or finished["completed"] != {"qwen": 200} or
        submission["gpus"] != 1 or submission["cases"] != 200 or submission["online_api_calls"] != 0):
        raise ValueError("A complete, valid, same-GPU study comparison is required")
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), model="Qwen3-4B-Instruct-2507",
        completed_cases=200, metric=report["metric"], reference_answers_used_only_posthoc=True,
        online_api_calls=0, same_gpu=True, teacher_probability_tensors_compared=False,
        teacher_and_cartridge_answer_formats_differ=True, development_used_for_selection=True,
        job_id=submission["job_id"], gpu=preflight["gpu"], probe_decision=probe_result,
        development_decision=decision, practice=report["practice"], overall=report["overall"],
        source_questions=len({row["case"].split("-", 2)[2] for row in rows}),
        group_source_questions={key: len({row["case"].split("-", 2)[2] for row in group})
                                for key, group in groups.items()}, groups=report["groups"])


def render(data):
    s, a, p = data["overall"], data["overall"]["accuracy"], data["practice"]
    lines = [
        "# Qwen：自练习提示的同 GPU 配对", "",
        "| 题数 | 全文 | 选中缓存零更新 | ICL Teacher | 原提示卡带 | 改进提示卡带 | 相对原提示 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {s['n']} | {a['whole']:.2f}% | {a['zero']:.2f}% | {a['teacher']:.2f}% | "
        f"{s['paired_baseline_accuracy']:.2f}% | {a['cartridge']:.2f}% | {s['strictly_paired']['delta_pp']:+.2f} 点 |", "",
        f"改进提示相对原提示新答对 {s['strictly_paired']['new_correct']} 题、"
        f"新答错 {s['strictly_paired']['new_wrong']} 题；与 Teacher 的差为 "
        f"{s['versus_teacher']['delta_pp']:+.2f} 点。两组卡带使用相同原问题格式，Teacher 附带通用回答格式要求。", "",
        f"相对同格式零更新，卡带新对 {s['versus_zero']['new_correct']} 题、新错 "
        f"{s['versus_zero']['new_wrong']} 题；相对 Teacher，新对 {s['versus_teacher']['new_correct']} 题、"
        f"新错 {s['versus_teacher']['new_wrong']} 题。", "",
        "| 题型 | 题数 | 来源问题 | 零更新 | Teacher | 卡带 | 相对零更新：新对 / 新错 | 相对 Teacher：新对 / 新错 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for number in range(1, 11):
        key = "task:qa" + str(number)
        if key not in data["groups"]:
            continue
        part = data["groups"][key]
        acc, zero, teacher = part["accuracy"], part["versus_zero"], part["versus_teacher"]
        lines.append(f"| QA{number} | {part['n']} | {data['group_source_questions'][key]} | "
                     f"{acc['zero']:.2f}% | {acc['teacher']:.2f}% | {acc['cartridge']:.2f}% | "
                     f"{zero['new_correct']} / {zero['new_wrong']} | "
                     f"{teacher['new_correct']} / {teacher['new_wrong']} |")
    lines += ["",
        f"10 题无标签预探针中，改进提示的已核验练习数由 {data['probe_decision']['base']} 增至 "
        f"{data['probe_decision']['candidate']}；但在完整 200 题中，两种提示分别产生 "
        f"{p['original']} 和 {p['study_only']} 个已核验练习，有 {p['more']} 题变多、{p['fewer']} 题变少。", "",
        "两组顺序运行在同一已加载模型和 GPU，使用相同选块、Teacher 生成 token 和零更新答案；"
        "每组均独立 prefill、训练、物化并在回答前释放全文及选中源缓存。完整 Teacher 概率张量未逐元素比较。", "",
        f"这是用于选参数的 200 题开发集，包含 {data['source_questions']} 个来源问题的四种长度。结果没有超过同轮 Teacher，"
        "因此按预定规则不扩展这一候选到 800 题。统一使用宽松词面评分，参考答案仅在生成后用于评分。", "",
        "[聚合数据](summary.json) · [复核脚本](../../../experiments/self_study_prompt/freeze_completed_study.py)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    names = ("manifest.json", "latest.json", "probe_verified.json", "preflight_qwen_0.json",
             "finished.json", "submission.json")
    data = freeze(*(json.loads((args.job_root / name).read_text()) for name in names))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(render(data))
    print("Qwen completed", data["completed_cases"], "development cases; candidate", data["overall"]["accuracy"]["cartridge"])


if __name__ == "__main__":
    main()
