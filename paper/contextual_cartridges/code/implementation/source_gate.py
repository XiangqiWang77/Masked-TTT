"""Deterministic structural Source-Support Gate recorded at Teacher generation.

The gate inspects one Teacher trajectory's binding and completeness only:
missing question or bound source, empty answer, truncated answer, or an
explicit refusal. It is not a factual verifier and never consults benchmark
answers or a second LLM judge. Semantic support stays unjudged; admitted
trajectories are candidates for the selective gate, nothing more.
"""
from __future__ import annotations

import re


def structural_source_gate(*, question, evidence, candidate, candidate_complete):
    """Conservative source binding/format checks, without a second LLM oracle."""
    if not question.strip() or not evidence.strip():
        reason = 'missing_question_or_bound_source'
    elif not candidate.strip():
        reason = 'empty_answer'
    elif not candidate_complete:
        reason = 'truncated_answer'
    elif re.fullmatch(
            r"\s*(?:sorry[,!]?\s*)?i (?:cannot|can't|am unable to) "
            r"(?:answer|help|assist)(?: with (?:this|that)(?: question| request)?)?[.!\s]*",
            candidate, flags=re.IGNORECASE):
        reason = 'explicit_refusal'
    else:
        reason = 'source_bound_complete_trajectory_semantic_support_unjudged'
    admitted = reason == 'source_bound_complete_trajectory_semantic_support_unjudged'
    return {'apply_update': admitted, 'reason': reason,
            'decision_source': 'structural_source_gate',
            'decision_rule': 'source binding and answer completeness only; the selective gate performs selection',
            'semantic_correctness_verified': False, 'llm_judge_used': False,
            'benchmark_answers_used': False}
