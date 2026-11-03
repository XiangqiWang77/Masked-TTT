"""Full-vocabulary teacher-to-student KL used by the joint update."""

import torch


def distribution_kl(logits, probabilities):
    logp = logits.log_softmax(-1)
    return (torch.special.xlogy(probabilities, probabilities) - probabilities * logp).sum(-1).mean()
