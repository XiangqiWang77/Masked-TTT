import pytest

from experiments.value_lr_physical.audit_olmo_context_overlap import paired


def test_paired_accuracy_and_gains_are_computed_on_the_same_rows():
    rows = [dict(teacher=0, cartridge=1), dict(teacher=1, cartridge=0),
            dict(teacher=0, cartridge=1), dict(teacher=1, cartridge=1)]
    assert paired(rows) == dict(n=4, baseline_accuracy=50, cartridge_accuracy=75,
                                new_correct=2, new_wrong=1, delta_points=25)


def test_paired_rejects_empty_or_nonbinary_data():
    with pytest.raises(ValueError, match="empty"):
        paired([])
    with pytest.raises(ValueError, match="binary"):
        paired([dict(teacher=2, cartridge=0)])
