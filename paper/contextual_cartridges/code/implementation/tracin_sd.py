"""Signed TracIn influence for native-KV selective self-distillation.

The reference objective is the complete ordinary-SD conversation set for one
request.  Candidate signals are source-admitted Teacher trajectories.  Scores
are computed only with gradients of the request-local Cartridge K,V; the
backbone remains frozen and no benchmark answer or held-out split is used.
"""
from __future__ import annotations

import math
import re

import torch


def structural_source_gate(*, question, evidence, candidate, candidate_complete):
    """Conservative source binding/format checks, without a second LLM oracle.

    Unknown semantic support remains eligible for TracIn. This is not a
    factual verifier, and a positive influence score is not a correctness label.
    """
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
            'decision_source': 'tracin_structural_source_gate',
            'decision_rule': 'source binding and answer completeness only; TracIn performs selection',
            'semantic_correctness_verified': False, 'llm_judge_used': False,
            'benchmark_answers_used': False}


def keep_score(score, threshold=0.0):
    """Strict threshold requested by the Selective-SD protocol."""
    score, threshold = float(score), float(threshold)
    if not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError("TracIn score and threshold must be finite")
    return score > threshold


def gradient_statistics(reference, candidate, *, learning_rate):
    """Return canonical signed TracIn dot product and an interpretable cosine."""
    if len(reference) != len(candidate) or not reference:
        raise ValueError("TracIn gradients must have the same nonzero parameter layout")
    dot = ref_sq = candidate_sq = 0.0
    for left, right in zip(reference, candidate):
        if left.shape != right.shape:
            raise ValueError("TracIn gradient shapes differ")
        left = left.detach().float()
        right = right.detach().float()
        dot += float((left * right).sum())
        ref_sq += float(left.square().sum())
        candidate_sq += float(right.square().sum())
    cosine = dot / (math.sqrt(ref_sq * candidate_sq) + 1e-12)
    return {
        "gradient_dot": dot,
        "gradient_cosine": cosine,
        "tracin_contribution": float(learning_rate) * dot,
        "reference_gradient_norm": math.sqrt(ref_sq),
        "candidate_gradient_norm": math.sqrt(candidate_sq),
    }


def _conversation_gradient(engine, cache, conversations, *, packed_tokens, seed):
    """Differentiate mean sparse-Teacher KL over one conversation collection."""
    from contextualized.early_study import distillation_terms, make_dataset
    dataset = make_dataset(engine, conversations, packed_tokens, seed)
    parameters = [parameter for parameter in cache.parameters()
                  if parameter.requires_grad]
    # Free each packed forward graph before starting the next one.  Keeping a
    # sum of differentiable losses here retains every activation graph and can
    # OOM even when ordinary SD fits comfortably on the same GPU.
    gradient_sums = [torch.zeros_like(parameter, device='cpu', dtype=torch.float32)
                     for parameter in parameters]
    numerator = 0.0
    count = 0
    for batch_index in range(len(dataset)):
        try:
            terms = distillation_terms(engine, cache, dataset[batch_index]).float()
            value = terms.sum()
            gradients = torch.autograd.grad(value, parameters, allow_unused=False)
            for accumulated, gradient in zip(gradient_sums, gradients):
                accumulated.add_(gradient.detach().to(device='cpu', dtype=torch.float32))
            numerator += float(value.detach())
            count += terms.numel()
            del gradients, value, terms
        finally:
            cache.clear()
    if count < 1:
        raise ValueError("TracIn objective has no sparse Teacher terms")
    for gradient in gradient_sums:
        gradient.div_(count)
    return tuple(gradient_sums), numerator / count, count


def score_checkpoint(engine, cache, reference_conversations,
                     candidate_conversations, *, packed_tokens, seed,
                     learning_rate, check=None, stream_reference=False):
    """Score candidate Teacher trajectories at one saved learning checkpoint."""
    if not reference_conversations or not candidate_conversations:
        return [], {"reference_loss": None, "candidates": 0}
    # Compute each trajectory once, independently of packing order.  A packed
    # reference may truncate a trajectory that is complete when scored alone;
    # subtracting its standalone gradient from that reference is not LOO.
    # Equal trajectory weights implement mean_{j != i} grad L_j exactly.
    reference = {}
    gradient_sum = None
    reference_loss = 0.0
    reference_terms = 0
    for index, conversation in enumerate(reference_conversations):
        if check:
            check()
        identity = id(conversation)
        if identity in reference:
            raise ValueError("A candidate occurs repeatedly in the reference objective")
        gradients, loss, terms = _conversation_gradient(
            engine, cache, [conversation], packed_tokens=packed_tokens,
            seed=seed + index)
        reference[identity] = (None if stream_reference else gradients, loss, terms, index)
        if gradient_sum is None:
            gradient_sum = tuple(gradient.clone() for gradient in gradients)
        else:
            for accumulated, gradient in zip(gradient_sum, gradients):
                accumulated.add_(gradient)
        reference_loss += loss
        reference_terms += terms
        if stream_reference:
            del gradients
    reference_loss /= len(reference)
    rows = []
    for index, conversation in enumerate(candidate_conversations):
        if id(conversation) not in reference:
            raise ValueError("A TracIn candidate must belong to the training reference")
        candidate_gradient, candidate_loss, candidate_terms, reference_index = reference[id(conversation)]
        remaining = len(reference) - 1
        if remaining < 1:
            # One signal cannot be interpreted relative to itself.
            rows.append({
                "candidate_index": index,
                "candidate_loss": candidate_loss,
                "tracin_contribution": -1e30,
                "gradient_dot": -1e30,
                "gradient_cosine": -1.0,
                "reference_gradient_norm": 0.0,
                "candidate_gradient_norm": 0.0,
                "leave_one_out_reference": True,
                "insufficient_other_reference_terms": True,
            })
            continue
        if stream_reference:
            if check:
                check()
            # Recompute one gradient at a time. This is twice the backward
            # compute but O(|KV|), rather than O(trajectories * |KV|), host RAM.
            candidate_gradient, _, _ = _conversation_gradient(
                engine, cache, [conversation], packed_tokens=packed_tokens,
                seed=seed + reference_index)
        comparison_gradient = tuple(
            (total - candidate) / remaining
            for total, candidate in zip(gradient_sum, candidate_gradient)
        )
        row = gradient_statistics(
            comparison_gradient, candidate_gradient,
            learning_rate=learning_rate)
        row.update(candidate_index=index, candidate_loss=candidate_loss,
                   candidate_terms=candidate_terms,
                   leave_one_out_reference=True,
                   insufficient_other_reference_terms=False)
        rows.append(row)
    del reference, gradient_sum
    return rows, {
        "reference_loss": reference_loss,
        "reference_terms": reference_terms,
        "reference_weighting": "equal_per_trajectory_loss",
        "reference_trajectories": len(reference_conversations),
        "gradient_storage": ("cpu_float32_two_pass_constant_memory" if stream_reference
                             else "cpu_float32_streamed_backward"),
        "candidates": len(candidate_conversations),
    }


def aggregate_checkpoints(checkpoint_rows, *, threshold=0.0):
    """Sum canonical TracIn contributions and apply one frozen threshold."""
    if not checkpoint_rows:
        raise ValueError("At least one TracIn checkpoint is required")
    width = len(checkpoint_rows[0])
    if any(len(rows) != width for rows in checkpoint_rows):
        raise ValueError("TracIn checkpoints scored different candidate sets")
    result = []
    for index in range(width):
        contributions = [rows[index]["tracin_contribution"]
                         for rows in checkpoint_rows]
        score = sum(contributions)
        result.append({
            "candidate_index": index,
            "tracin_score": score,
            "checkpoint_contributions": contributions,
            "threshold": float(threshold),
            "keep": keep_score(score, threshold),
            "interpretation": "proponent" if score > threshold else "opponent_or_nonpositive",
        })
    return result
