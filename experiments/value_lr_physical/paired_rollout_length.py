#!/usr/bin/env python3
"""Audit completed 8/32-token on-policy runs with matching evidence and prompts.

Reads saved generated results and already-computed binary scores. It performs
no new model inference, selection, API call, or access to reference answers.
Only aggregates, never raw prompts, predictions, or case IDs, are written.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median


PRACTICE_FIELDS = ("question", "answer", "evidence", "source_chunk", "verifier_answer", "accepted")


def _practices(result):
    return [{key: item.get(key) for key in PRACTICE_FIELDS}
            for item in result["self_study"]["checks"]]


def pair(case, old, new, score_old, score_new):
    """Return aggregate-ready data and mismatch reasons for one saved pair."""
    if (old["key"] != case or new["key"] != case or score_old["case"] != case or
        score_new["case"] != case or old["task"] != score_old["task"] or
        new["task"] != score_new["task"]):
        raise ValueError("Saved result and score identity differ")
    a, b = old["training"], new["training"]
    for training, limit in ((a, 8), (b, 32)):
        if (training["objective"] != "on-policy KL(teacher || student)" or
            training["student_rollout_limit"] != limit or training["value_basis_source"] != "document" or
            training["selected_transfer_weight"] != 0 or not training["teacher_conditioned_on_student_prefix"] or
            not training["student_rollouts_refreshed_each_step"] or not training["steps"]):
            raise ValueError("Unexpected original-SD training protocol")
    if any(result["self_study"]["reference_answers_used"] or not result["backbone_unchanged"] or
           not result["original_document_KV_unchanged"] for result in (old, new)):
        raise ValueError("The backbone, cache or unlabeled study contract changed")
    first, longer = a["steps"][0], b["steps"][0]
    for step in (first, longer):
        if (step["question_path"] != "main" or not step["student_rollout_tokens"] or
            step["target_positions"] != list(range(len(step["student_rollout_tokens"]))) or
            not all(math.isfinite(step[k]) for k in ("teacher_KL_before", "gain_gradient_norm", "value_gradient_norm")) or
            any(step[k] < 0 for k in ("gain_gradient_norm", "value_gradient_norm"))):
            raise ValueError("Unexpected initial causal SD step")
    original = old["standalone_cartridge"]["paired_prompt_control"]
    control = new["standalone_cartridge"]["original_question_controls"]
    probe = new["standalone_cartridge"]["step_probe"]
    checkpoint = probe["checkpoints"]["2"]
    if (original["reference_answers_used"] or original["teacher_answer_in_student_prompt"] or
        control["reference_answers_used"] or control["teacher_answer_in_student_prompt"] or
        probe["reference_answers_used"] or probe["teacher_answer_in_final_prompt"] or
        not original["same_selected_kv"] or not original["same_generation_limit"] or
        not control["same_selected_kv"] or not control["same_generation_limit"] or
        not probe["same_selected_kv"] or probe["full_kv_used_at_decode"] or
        checkpoint["total_steps"] != 2 + new["self_study"]["verified"] or
        old["standalone_cartridge"]["generated_token_ids"] != original["trained_original"]["tokens"] or
        old["standalone_cartridge"]["prediction"] != original["trained_original"]["prediction"]):
        raise ValueError("The same selected-KV two-step control is required")
    checks = {
        "selection": old["selection"]["selected_original_chunks"] == new["selection"]["selected_original_chunks"],
        "teacher_tokens": old["teacher"]["tokens"] == new["teacher"]["tokens"],
        "zero_tokens": original["zero_original"]["tokens"] == control["zero"]["tokens"],
        "practices": _practices(old) == _practices(new),
        "student_prefix": longer["student_rollout_tokens"][:len(first["student_rollout_tokens"])] == first["student_rollout_tokens"],
        "task_and_length": (score_old["task"], score_old["length"]) == (score_new["task"], score_new["length"]),
        "control_scores": (score_old["teacher"], score_old["zero"]) == (score_new["teacher"], score_new["zero"]),
    }
    mismatch = [key for key, ok in checks.items() if not ok]
    if mismatch:
        return None, mismatch
    short_score, long_score = score_old["cartridge"], score_new["step2"]
    if type(short_score) is not int or type(long_score) is not int or short_score not in (0, 1) or long_score not in (0, 1):
        raise ValueError("Expected binary post-generation quality scores")
    row = dict(case=case, task=score_old["task"], short=short_score, long=long_score,
        first_kl=first["teacher_KL_before"], longer_kl=longer["teacher_KL_before"],
        short_gain_gradient=first["gain_gradient_norm"], long_gain_gradient=longer["gain_gradient_norm"],
        short_value_gradient=first["value_gradient_norm"], long_value_gradient=longer["value_gradient_norm"],
        lengthened=len(longer["student_rollout_tokens"]) > len(first["student_rollout_tokens"]),
        output_tokens_changed=original["trained_original"]["tokens"] != checkpoint["tokens"])
    return row, []


def summarize(rows):
    if not rows:
        return dict(n=0, source_questions=0)
    result = dict(n=len(rows), source_questions=len({r["case"].split("-", 2)[2] for r in rows}),
        lengthened=sum(r["lengthened"] for r in rows),
        output_tokens_changed=sum(r["output_tokens_changed"] for r in rows),
        short_accuracy=100 * sum(r["short"] for r in rows) / len(rows),
        long_accuracy=100 * sum(r["long"] for r in rows) / len(rows),
        new_correct=sum(not r["short"] and r["long"] for r in rows),
        new_wrong=sum(r["short"] and not r["long"] for r in rows))
    for name in ("first_kl", "longer_kl", "short_gain_gradient", "long_gain_gradient",
                 "short_value_gradient", "long_value_gradient"):
        result["median_" + name] = median(r[name] for r in rows)
    return result


def collect(old_root, new_root):
    old_manifest = json.loads((old_root / "manifest.json").read_text())
    new_manifest = json.loads((new_root / "manifest.json").read_text())
    old_report = json.loads((old_root / "latest.json").read_text())
    new_report = json.loads((new_root / "latest.json").read_text())
    ids = old_manifest["case_ids"]
    for manifest in (old_manifest, new_manifest):
        if (manifest["case_count"] != 200 or manifest["case_ids"] != ids or
            manifest["model_protocols"] != old_manifest["model_protocols"] or
            manifest["selection_policy"] != "decision_entity_link_v1" or
            manifest["entity_reserve_fraction"] != .25 or manifest["keep_fraction"] != .1 or
            manifest["basis_source"] != "document" or manifest["selected_transfer_weight"] != 0 or
            manifest["online_api_calls"] != 0 or manifest["reference_answers_used_in_inference"] is not False):
            raise ValueError("The fixed 200-case evaluation protocols differ")
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("Expected 200 distinct development cases")
    for report in (old_report, new_report):
        if (report["metric"] != "answer_focused_lexical_v2" or report["expected_per_model"] != 200 or
            report["online_api_calls"] != 0 or report["errors"] or report["invalid"]):
            raise ValueError("Both scored development reports must be complete")
    results = {}
    for model in ("qwen", "gpt"):
        scored = []
        for report in (old_report, new_report):
            rows = report["models"][model]["rows"]
            lookup = {row["case"]: row for row in rows}
            if len(rows) != 200 or len(lookup) != 200 or set(lookup) != set(ids):
                raise ValueError("Missing, duplicated or unexpected paired scores")
            scored.append(lookup)
        pairs, rejected = [], Counter()
        for case in ids:
            old = json.loads((old_root / "cases" / model / case / "result.json").read_text())
            new = json.loads((new_root / "cases" / model / case / "result.json").read_text())
            row, reasons = pair(case, old, new, scored[0][case], scored[1][case])
            if row is None:
                rejected.update(reasons)
            else:
                pairs.append(row)
        groups = defaultdict(list)
        for row in pairs:
            groups["task:" + row["task"]].append(row)
            groups["first_kl:" + ("below_1e-6" if row["first_kl"] < 1e-6 else "at_least_1e-6")].append(row)
        results[model] = dict(matched=summarize(pairs), rejected_case_count=200-len(pairs),
                              mismatch_reasons=dict(sorted(rejected.items())),
                              groups={key: summarize(value) for key, value in sorted(groups.items())})
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), model_calls=0, online_api_calls=0,
        reference_answers_read=False, metric="answer_focused_lexical_v2", development_used_for_selection=True,
        icl_teacher_logits_not_saved=True, same_gpu_not_guaranteed=True,
        selected_cache_diagnostic_before_source_release=True, models=results)


def render(data):
    def number(x):
        return "—" if x is None else f"{x:.4g}"
    lines = ["# 8 与 32 token：同题蒸馏信号与卡带回答", "",
        "| 模型 / 范围 | 配对题数 | 来源问题 | 实际加长 | 首步 KL 中位数：8 → 32 | Value 梯度中位数：8 → 32 | 答案 token 改变 | 准确率：8 → 32 | 新对 / 新错 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, item in data["models"].items():
        for label, stats in (("全部严格配对", item["matched"]), ("QA3", item["groups"].get("task:qa3", {"n": 0}))):
            if not stats["n"]:
                continue
            lines.append(f"| {model} {label} | {stats['n']} | {stats['source_questions']} | {stats['lengthened']} | "
                f"{number(stats['median_first_kl'])} → {number(stats['median_longer_kl'])} | "
                f"{number(stats['median_short_value_gradient'])} → {number(stats['median_long_value_gradient'])} | "
                f"{stats['output_tokens_changed']} | {stats['short_accuracy']:.2f}% → {stats['long_accuracy']:.2f}% | "
                f"{stats['new_correct']} / {stats['new_wrong']} |")
    lines += ["", "每个模型各 200 题开发集；表内仅保留选块、Teacher 生成 token、零更新 token、自练习及学生前 8 个 token 一致的题。"
        "训练目标均为全文 Student 对独立选中 Teacher 的 on-policy KL，表中比较相同两步预算的选中缓存输出。",
        "", "两侧的两步输出都是释放选中源缓存前的配对诊断，两轮可能使用不同 GPU，也未保存 Teacher 的完整概率张量；"
        "因此匹配的是可见运行条件，不能称为同一次运行。更长前缀的首步 KL 均值涵盖更多位置，不能把大小直接解释成同位置改善。",
        "", "QA3 组的梯度增加而答案不变，说明这批 QA3 题里仅延长训练前缀未带来可观察的答案变化；它不能单独确定变化损失在全文到卡带的哪一环。"
        "完整数据均已完成，本审计未重新调用模型或 API；二值准确率来自先前生成后的统一宽松词面评分。", "",
        "[聚合数据](summary.json) · [复核脚本](../../../experiments/value_lr_physical/paired_rollout_length.py)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.workspace / "query_cartridges_linked_selection_20261104",
                   args.workspace / "query_cartridges_longsteps_200dev_20261104")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(render(data))
    for model, stats in data["models"].items():
        print(model, "matched", stats["matched"]["n"], "QA3", stats["groups"].get("task:qa3", {}).get("n", 0))


if __name__ == "__main__":
    main()
