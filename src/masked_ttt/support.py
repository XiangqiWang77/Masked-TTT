"""Admit a teacher signal using source evidence, before any memory update."""

from .admission import SelectiveSDPolicy, decide_update
from .source_evidence import EvidenceJudgeOptions, independent_evidence_judge
from .support_verification import task_conditioned_evidence_judge
from .support_views import SelfJudgeOptions, judge_before_update, selected_evidence_text


def check_signal(backbone, reader, cache, ids, request, layout, selected, teacher):
    options = EvidenceJudgeOptions(
        extraction_tokens=512, verification_tokens=384, maximum_quotes=12, maximum_occurrences=64
    )
    views = judge_before_update(
        backbone,
        reader,
        cache,
        layout.document_end,
        ids[: layout.document_end].tolist(),
        request.prompt,
        request.document,
        teacher.prediction,
        SelfJudgeOptions(
            max_new_tokens=384,
            policy="grounded_veto",
            focused_max_new_tokens=256,
            focused_hint_policy="verified_quotes_only",
        ),
        question=request.question,
        selected_evidence=selected_evidence_text(backbone.tokenizer, ids, layout.chunks, selected),
        candidate_complete=teacher.complete,
    )
    extracted = independent_evidence_judge(
        backbone,
        reader,
        cache,
        ids,
        layout.document_end,
        request.prompt,
        request.document,
        request.question,
        teacher.prediction,
        views,
        options,
        skip_generic_verification=True,
    )
    assessment = task_conditioned_evidence_judge(
        backbone,
        reader,
        request.prompt,
        request.document,
        request.question,
        teacher.prediction,
        extracted,
        options,
    )
    decision = decide_update(assessment, SelectiveSDPolicy(backbone.config.steps, 0))
    return dict(
        decision=decision,
        focused_and_full_views=views,
        candidate_blind_extraction=extracted,
        final_assessment=assessment,
    )
