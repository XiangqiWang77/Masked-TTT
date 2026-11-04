import pytest

from experiments.anchored_short_verifier.audit_qa3_answer_transfer import summarize


def test_answer_transfer_summary_preserves_paired_changes_and_source_count():
    rows = [dict(case="babilong-8-qa3-1", native_old=0, adapted_old=0, native_new=0,
                 adapted_new=1, zero=0, cartridge=1, teacher=0, old_full_changed=False,
                 new_full_changed=True, full_8_32_changed=True, selected_8_32_changed=True,
                 selected_update_changed=True),
            dict(case="babilong-16-qa3-1", native_old=1, adapted_old=1, native_new=1,
                 adapted_new=0, zero=1, cartridge=0, teacher=1, old_full_changed=False,
                 new_full_changed=True, full_8_32_changed=True, selected_8_32_changed=False,
                 selected_update_changed=True)]
    result = summarize(rows)
    assert result["n"] == 2 and result["source_questions"] == 1
    assert result["pairs"]["zero_to_cartridge"] == dict(new_correct=1, new_wrong=1)
    assert result["output_changes"]["selected_8_32_changed"] == 1
    rows[0]["cartridge"] = 2
    with pytest.raises(ValueError):
        summarize(rows)
