from experiments.value_lr_physical.audit_loose_scorer import qa8_possible_missed_set


def test_exact_noun_set_is_only_a_review_flag():
    assert qa8_possible_missed_set("Mary carries the apple and milk", ["apple,milk"], 0)
    assert not qa8_possible_missed_set("Mary carries the apple and milk", ["apple,milk"], 1)
    assert not qa8_possible_missed_set("Mary carries the apple or milk", ["apple,milk"], 0)
    assert not qa8_possible_missed_set("Mary carries apple but not milk", ["apple,milk"], 0)
    assert not qa8_possible_missed_set("apple, milk, football", ["apple,milk"], 0)
    assert not qa8_possible_missed_set("nothing", ["nothing"], 0)
