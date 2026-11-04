from copy import deepcopy

import pytest

from experiments.babilong_linked_selection.qwen_full_support_partition import groups, outcome, summarize


def row(coverage, whole, teacher, zero, cartridge, source="qa1-1"):
    return dict(coverage=coverage, whole=whole, teacher=teacher, zero=zero, cartridge=cartridge,
                zero_teacher_format=teacher, cartridge_teacher_format=cartridge,
                source=source, task="qa1", length=8192)


def test_support_coverage_and_full_teacher_outcome_are_distinct_axes():
    rows = [row("complete", 1, 0, 0, 1), row("incomplete", 1, 0, 0, 0),
            row("complete", 0, 1, 1, 1, source="qa1-2"), row("ambiguous", 1, 1, 1, 0)]
    stats = summarize(rows)
    assert stats["n"] == 4 and stats["source_questions"] == 2
    assert stats["coverage_counts"] == dict(complete=2, incomplete=1, ambiguous=1, unavailable=0)
    assert stats["answer_groups"] == dict(both_correct=1, full_only=2, teacher_only=1, both_wrong=0)
    assert stats["pairs"]["teacher"] == dict(new_correct=1, new_wrong=1, delta_pp=0.)
    assert outcome(rows[0]) == "full_only"
    split = groups(rows)
    assert split["full_teacher:full_only"]["coverage_counts"]["complete"] == 1
    assert split["full_teacher:full_only"]["coverage_counts"]["incomplete"] == 1
    assert split["coverage:complete"]["n"] == 2
    assert split["task:qa1/full_only"]["n"] == 2
    assert split["length:8192/full_only"]["coverage_counts"]["incomplete"] == 1
    assert split["coverage:unavailable"]["accuracy"]["whole"] is None


def test_nonbinary_answer_and_unknown_coverage_are_rejected():
    example = row("complete", 1, 0, 0, 1)
    bad = deepcopy(example); bad["teacher"] = 2
    with pytest.raises(ValueError, match="Invalid"):
        summarize([bad])
    bad = deepcopy(example); bad["coverage"] = "probably"
    with pytest.raises(ValueError, match="Invalid"):
        summarize([bad])
