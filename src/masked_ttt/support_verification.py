"""Recheck fixed source excerpts with the original task instructions present.

This development critic preserves an earlier candidate-blind extraction. Only
the fresh verification prompt changes; it never rewrites or trains a teacher.
"""

from copy import deepcopy
import hashlib
import json

from .source_evidence import (
    VERIFICATION,
    parse_verification,
    verified_occurrences,
)


def task_conditioned_evidence_judge(
    engine, reader, prompt, document, question, candidate, parent_judgment, options
):
    import torch
    from transformers.cache_utils import DynamicCache
    from .support_views import _generate_assessment

    if not document or prompt.count(document) != 1:
        raise ValueError("Require exactly one original document in the task prompt")
    if not isinstance(question, str) or not question.strip() or question not in prompt:
        raise ValueError("Preserve the original question in its task prompt")
    if (
        not parent_judgment["independent_extraction_had_no_candidate"]
        or parent_judgment["old_verdict_or_reason_supplied_to_model"]
        or parent_judgment["teacher_rewritten"]
        or parent_judgment["reference_answers_used"]
        or type(parent_judgment["old_decision"]) is not bool
    ):
        raise ValueError("Require the original candidate-blind source assessment")
    occurrences = deepcopy(parent_judgment["verified_source_occurrences"])
    verified, ambiguity = verified_occurrences(
        [item["quote"] for item in occurrences], document, options.maximum_occurrences
    )
    if ambiguity or verified != occurrences:
        raise ValueError("The fixed source occurrences changed or lack source support")

    start = prompt.index(document)
    task = (
        prompt[:start]
        + "[Original source excerpts for this evidence check appear below.]"
        + prompt[start + len(document) :]
    )
    assessment = dict(
        resolved=False,
        apply_update=None,
        parse_valid=False,
        reason="no_source_verified_evidence",
        evidence=[],
        source_quotes_verified=False,
    )
    generation = None
    prompt_sha = None
    if occurrences:
        context = (
            "Original task instructions and question:\n"
            + task
            + "\n\nVerbatim source excerpts, in original document order:\n"
            + json.dumps(occurrences, ensure_ascii=False)
            + "\n\n"
            + VERIFICATION
            + "\nOriginal question (JSON):\n"
            + json.dumps(question, ensure_ascii=False)
            + "\nUnchanged candidate (JSON):\n"
            + json.dumps(candidate, ensure_ascii=False)
        )
        rendered = engine.tokenizer.apply_chat_template(
            [dict(role="user", content=context)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = engine.tokenizer.encode(rendered, add_special_tokens=False)
        prompt_sha = hashlib.sha256(rendered.encode()).hexdigest()
        with torch.no_grad():
            generation = _generate_assessment(
                engine,
                reader,
                DynamicCache(config=engine.model.config),
                inputs,
                0,
                options.verification_tokens,
            )
        assessment = parse_verification(
            generation["prediction"], document, complete=generation["complete"]
        )

    return dict(
        assessment,
        extraction=deepcopy(parent_judgment["extraction"]),
        extraction_parse_valid=parent_judgment["extraction_parse_valid"],
        extraction_prompt_sha256=parent_judgment["extraction_prompt_sha256"],
        independent_extraction_had_no_candidate=True,
        verified_source_occurrences=occurrences,
        extraction_reused_exactly=True,
        parent_extraction_context_view=parent_judgment.get(
            "extraction_context_view", "full_document"
        ),
        verification=generation,
        verification_prompt_sha256=prompt_sha,
        verification_context_view="original_task_plus_fixed_source_excerpts",
        old_verdict_or_reason_supplied_to_model=False,
        old_decision=parent_judgment["old_decision"],
        proposed_decision=(
            assessment["apply_update"]
            if assessment["resolved"]
            else parent_judgment["old_decision"]
        ),
        decision_rule="source-resolved task-conditioned verification; otherwise original gate",
        teacher_rewritten=False,
        reference_answers_used=False,
        original_document_KV_reused=False,
        original_document_KV_immutable=True,
    )
