import pytest

from experiments.value_lr_physical.audit_olmo_context_overlap import complementarity, paired


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


def test_recovery_and_retention_use_the_correct_view_before_training():
    rows = [dict(whole=1, zero=0, trained=1), dict(whole=1, zero=0, trained=0),
            dict(whole=0, zero=1, trained=1), dict(whole=0, zero=1, trained=0),
            dict(whole=1, zero=1, trained=1)]
    assert complementarity(rows) == dict(n=5, whole_only=2, whole_only_recovered=1,
                                          selected_only=2, selected_only_retained=1)
