import pytest

from experiments.babilong_linked_selection.qwen_practice_outcomes import summarize


def row(source, zero, cartridge, zero_teacher_format, cartridge_teacher_format, verified=0):
    return dict(source=source, verified=verified, rejections={"invalid_json": int(verified == 0)},
                whole=0, teacher=1, zero=zero, cartridge=cartridge,
                zero_teacher_format=zero_teacher_format, cartridge_teacher_format=cartridge_teacher_format)


def test_compares_both_suffix_formats_in_matched_pairs():
    rows = [row("a", 0, 1, 1, 0), row("b", 1, 0, 0, 1, 1),
            row("a", 0, 1, 0, 1, 1), row("c", 0, 0, 1, 1)]
    got = summarize(rows)
    assert got["n"] == 4 and got["source_questions"] == 3 and got["verified_at_least_one"] == 2
    assert got["pairs"]["zero"] == dict(new_correct=2, new_wrong=1, delta_pp=25)
    assert got["pairs"]["zero_teacher_format"] == dict(new_correct=2, new_wrong=1, delta_pp=25)
    assert got["rejection_counts"] == {"invalid_json": 2}


def test_empty_and_nonbinary_outcomes():
    assert summarize([])["pairs"]["zero"]["delta_pp"] is None
    bad = row("a", 1, 1, 1, 1)
    bad["zero"] = True
    with pytest.raises(ValueError, match="binary"):
        summarize([bad])
    bad["zero"] = 1
    bad["verified"] = 3
    with pytest.raises(ValueError, match="verified"):
        summarize([bad])
