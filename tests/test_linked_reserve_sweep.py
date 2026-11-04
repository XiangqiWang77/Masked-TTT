from experiments.babilong_linked_selection.audit_reserve_sweep import summarize


def test_reserve_sweep_counts_only_observed_coverage_transitions():
    def row(before, after, old_ids, new_ids):
        return {
            "total_chunks": 20,
            "outcomes": {
                0.0: {"status": before, "selected": old_ids},
                0.25: {"status": after, "selected": new_ids},
            },
        }

    rows = [
        row("incomplete", "complete", [1, 2], [1, 3]),
        row("complete", "incomplete", [1, 2], [2, 3]),
        row("ambiguous", "complete", [1, 2], [1, 2]),
        row("complete", "complete", [1, 2], [1, 2]),
    ]
    summary = summarize(rows, 0.25)
    assert summary == {
        "cases": 4,
        "complete": 3,
        "incomplete": 1,
        "ambiguous": 0,
        "unavailable": 0,
        "new_complete": 1,
        "lost_complete": 1,
        "changed_selection": 2,
        "selected_chunks": 8,
        "total_chunks": 80,
    }
    baseline = summarize(rows, 0.0)
    assert baseline["complete"] == 2
    assert baseline["ambiguous"] == 1
    assert baseline["new_complete"] == baseline["lost_complete"] == 0
