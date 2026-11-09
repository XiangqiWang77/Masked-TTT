"""Signed TracIn influence for native-KV selective self-distillation.

The reference objective is the complete ordinary-SD conversation set for one
request.  Candidate signals are source-admitted Teacher trajectories.  Scores
are computed only with gradients of the request-local Cartridge K,V; the
backbone remains frozen and no benchmark answer or held-out split is used.
"""
from __future__ import annotations

import math

import torch


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
    numerator = None
    count = 0
    for batch_index in range(len(dataset)):
        terms = distillation_terms(engine, cache, dataset[batch_index]).float()
        value = terms.sum()
        numerator = value if numerator is None else numerator + value
        count += terms.numel()
        cache.clear()
    if numerator is None or count < 1:
        raise ValueError("TracIn objective has no sparse Teacher terms")
    loss = numerator / count
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    cache.clear()
    return (tuple(gradient.detach() for gradient in gradients),
            float(loss.detach()), count)


def score_checkpoint(engine, cache, reference_conversations,
                     candidate_conversations, *, packed_tokens, seed,
                     learning_rate):
    """Score candidate Teacher trajectories at one saved learning checkpoint."""
    if not reference_conversations or not candidate_conversations:
        return [], {"reference_loss": None, "candidates": 0}
    reference_gradient, reference_loss, reference_terms = _conversation_gradient(
        engine, cache, reference_conversations,
        packed_tokens=packed_tokens, seed=seed)
    rows = []
    for index, conversation in enumerate(candidate_conversations):
        candidate_gradient, candidate_loss, candidate_terms = _conversation_gradient(
            engine, cache, [conversation], packed_tokens=packed_tokens,
            seed=seed + 1 + index)
        occurrences = sum(value is conversation for value in reference_conversations)
        if occurrences > 1:
            raise ValueError("A candidate occurs repeatedly in the reference objective")
        if occurrences == 1:
            remaining = reference_terms - candidate_terms
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
            comparison_gradient = tuple(
                (full * reference_terms - candidate * candidate_terms) / remaining
                for full, candidate in zip(reference_gradient, candidate_gradient)
            )
        else:
            comparison_gradient = reference_gradient
        row = gradient_statistics(
            comparison_gradient, candidate_gradient,
            learning_rate=learning_rate)
        row.update(candidate_index=index, candidate_loss=candidate_loss,
                   candidate_terms=candidate_terms,
                   leave_one_out_reference=occurrences == 1,
                   insufficient_other_reference_terms=False)
        rows.append(row)
    del reference_gradient
    return rows, {
        "reference_loss": reference_loss,
        "reference_terms": reference_terms,
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
