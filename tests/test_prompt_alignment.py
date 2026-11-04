from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from masked_ttt import MaskedTTT, Request, TTTConfig
from masked_ttt.prompt_alignment import main_student_suffix
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
import masked_ttt.pipeline as pipeline
from test_selected_transfer import gpt_engine
from test_standalone import make_engine


def test_only_question_tokens_selected_and_context_checked():
    original = torch.tensor([1, 2])
    teacher = SimpleNamespace(question_tokens=torch.tensor([1, 3, 2]),
                              tokens=torch.tensor([999]))
    assert torch.equal(main_student_suffix(original, teacher, align=False,
        document_length=5, max_context=20, rollout_tokens=2), original)
    actual = main_student_suffix(original, teacher, align=True,
        document_length=5, max_context=20, rollout_tokens=2)
    assert torch.equal(actual, teacher.question_tokens)
    assert 999 not in actual.tolist()
    with pytest.raises(ValueError, match="exceed"):
        main_student_suffix(original, teacher, align=True,
            document_length=5, max_context=9, rollout_tokens=2)
    with pytest.raises(ValueError, match="boolean"):
        TTTConfig(align_student_teacher_prompt="yes")


@pytest.mark.parametrize("architecture", ["qwen3", "gpt_oss"])
@pytest.mark.parametrize("align", [False, True])
def test_on_policy_student_uses_intended_question_suffix(monkeypatch, architecture, align):
    engine = make_engine() if architecture == "qwen3" else gpt_engine()
    engine.config = replace(engine.config, align_student_teacher_prompt=align,
        self_question_count=0, selected_transfer_weight=0.0,
        standalone_answer_format="teacher_requirements")
    request = Request(document="Old report says red. New report says blue. " * 6,
                      question="What is the new report?", max_new_tokens=3)

    class Selector:
        def prepare(self, backbone, original):
            layout = prepare_layout(backbone, original)
            return PreparedRequest(original, layout, [1, len(layout.chunks)-2],
                                   dict(backend="unit_test"))

    student_suffixes, teacher_suffixes = [], []
    old_rollout, old_score = pipeline.student_rollout, pipeline.score_student_prefix

    def record_rollout(_engine, reader, cache, suffix, limit):
        student_suffixes.append(suffix.clone())
        return old_rollout(_engine, reader, cache, suffix, limit)

    def record_score(_engine, reader, teacher, tokens):
        teacher_suffixes.append(teacher.question_tokens.clone())
        return old_score(_engine, reader, teacher, tokens)

    monkeypatch.setattr(pipeline, "student_rollout", record_rollout)
    monkeypatch.setattr(pipeline, "score_student_prefix", record_score)
    result = MaskedTTT(engine, Selector()).answer(request)
    assert len(student_suffixes) == len(teacher_suffixes) == engine.config.steps
    assert all(torch.equal(student, teacher) == align
               for student, teacher in zip(student_suffixes, teacher_suffixes))
    assert result["training"]["student_question_matches_teacher"] == align
    assert result["training"]["objective"] == "on-policy KL(teacher || student)"
    assert result["training"]["selected_transfer_weight"] == 0
    assert any(step["gain_gradient_norm"] > 0 for step in result["training"]["steps"])
    assert any(step["value_gradient_norm"] > 0 for step in result["training"]["steps"])
    assert result["original_document_KV_unchanged"] and result["backbone_unchanged"]
    assert not result["teacher_answer_in_final_prompt"]
