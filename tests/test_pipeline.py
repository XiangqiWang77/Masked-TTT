import json

import pytest
import torch

from masked_ttt import MaskedTTT, Request
from masked_ttt.admission import SelectiveSDPolicy, decide_update
from masked_ttt.optimizer import distribution_kl
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from masked_ttt.source_evidence import parse_verification


class FixedSelector:
    def prepare(self, backbone, request):
        layout = prepare_layout(backbone, request)
        return PreparedRequest(request, layout, [1, 3], dict(complete_original_document=True))


@pytest.mark.parametrize("admit", [False, True])
def test_end_to_end_request_isolated_updates_and_fresh_final_question(
    tiny_backbone, monkeypatch, admit
):
    import masked_ttt.pipeline as module

    question = "Which budget was finally approved?"
    request = Request(
        document="X had a draft budget of 300. X finally received 320. " * 3,
        question=question,
        max_new_tokens=3,
    )
    planned = tiny_backbone.config.steps if admit else 0
    monkeypatch.setattr(
        module,
        "check_signal",
        lambda *a, **kw: dict(
            decision=dict(
                apply_update=admit,
                planned_steps=planned,
                state="supported" if admit else "unresolved",
            )
        ),
    )
    calls = []
    original_forward = module.private_forward

    def observe(reader, tokens, start, branch):
        calls.append((tokens.clone(), start))
        return original_forward(reader, tokens, start, branch)

    monkeypatch.setattr(module, "private_forward", observe)
    before = [p.detach().clone() for p in tiny_backbone.model.parameters()]
    method = MaskedTTT(tiny_backbone, FixedSelector())
    result = method.answer(request)
    json.dumps(result, allow_nan=False)
    assert len(result["training"]["steps"]) == planned
    assert result["training"]["final_kl"] <= result["training"]["initial_kl"] + 1e-7
    assert result["final_question_recomputed"] and not result["teacher_answer_in_final_prompt"]
    layout = prepare_layout(tiny_backbone, request)
    torch.testing.assert_close(
        calls[1][0], torch.tensor(layout.ids[layout.document_end :]), atol=0, rtol=0
    )
    assert calls[1][1] == layout.document_end
    assert result["timing"]["ttft_seconds"] > 0
    for parameter, previous in zip(tiny_backbone.model.parameters(), before):
        torch.testing.assert_close(parameter, previous, atol=0, rtol=0)
        assert parameter.grad is None and not parameter.requires_grad
    assert not hasattr(method, "memory")
    if admit:
        assert any(
            r["g_gradient_norm"] > 0 and r["V_gradient_norm"] > 0
            for r in result["training"]["steps"]
        )
    else:
        assert result["training"]["accepted_steps"] == 0


def test_soft_KL_uses_non_argmax_probability_mass():
    logits = torch.tensor([[0.2, 0.5, -0.1]], requires_grad=True)
    teacher = torch.tensor([[0.70, 0.25, 0.05]])
    loss = distribution_kl(logits, teacher)
    (gradient,) = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, logits.softmax(-1) - teacher)
    hard_target = torch.tensor([[1.0, 0.0, 0.0]])
    (hard,) = torch.autograd.grad(distribution_kl(logits, hard_target), logits)
    assert not torch.equal(gradient, hard)


@pytest.mark.parametrize(
    "supported,contradicted,expected", [(True, False, 16), (False, True, 0), (False, False, 0)]
)
def test_source_support_controls_admission_without_gold(supported, contradicted, expected):
    document = "X finally received 320."
    content = dict(
        reason="Check the source.",
        evidence=[document],
        candidate_has_answer=True,
        candidate_supported=supported,
        candidate_contradicted=contradicted,
    )
    assessment = parse_verification(json.dumps(content), document)
    decision = decide_update(assessment, SelectiveSDPolicy())
    assert decision["planned_steps"] == expected


def test_fabricated_support_quote_does_not_admit_a_signal():
    content = dict(
        reason="An unsupported assertion.",
        evidence=["X received 999."],
        candidate_has_answer=True,
        candidate_supported=True,
        candidate_contradicted=False,
    )
    assessment = parse_verification(json.dumps(content), "X finally received 320.")
    assert decide_update(assessment, SelectiveSDPolicy())["planned_steps"] == 0
