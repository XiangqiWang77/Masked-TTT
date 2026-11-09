import pytest
import torch

from contextualized.tracin_sd import (
    aggregate_checkpoints,
    gradient_statistics,
    keep_score,
)


def test_signed_gradient_dot_identifies_proponent_and_opponent():
    reference = (torch.tensor([1.0, 2.0]),)
    proponent = gradient_statistics(
        reference, (torch.tensor([2.0, 1.0]),), learning_rate=0.1)
    opponent = gradient_statistics(
        reference, (torch.tensor([-2.0, -1.0]),), learning_rate=0.1)
    assert proponent["tracin_contribution"] > 0
    assert opponent["tracin_contribution"] < 0


def test_strict_threshold_keeps_only_positive_influence():
    assert keep_score(0.01, 0.0) is True
    assert keep_score(0.0, 0.0) is False
    assert keep_score(-0.01, 0.0) is False
    with pytest.raises(ValueError):
        keep_score(float("nan"), 0.0)


def test_tracin_sums_contributions_across_learning_checkpoints():
    rows = aggregate_checkpoints(
        [
            [
                {"tracin_contribution": 2.0},
                {"tracin_contribution": -2.0},
            ],
            [
                {"tracin_contribution": -0.5},
                {"tracin_contribution": 0.5},
            ],
        ],
        threshold=0.0,
    )
    assert rows[0]["tracin_score"] == 1.5
    assert rows[0]["keep"] is True
    assert rows[1]["tracin_score"] == -1.5
    assert rows[1]["keep"] is False


def test_tracin_rejects_mismatched_gradient_layout():
    with pytest.raises(ValueError):
        gradient_statistics(
            (torch.tensor([1.0]),),
            (torch.tensor([1.0]), torch.tensor([2.0])),
            learning_rate=0.1,
        )
