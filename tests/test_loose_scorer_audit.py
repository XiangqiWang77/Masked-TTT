from experiments.value_lr_physical.audit_loose_scorer import qa7_count_screen, qa8_possible_missed_set


def test_exact_noun_set_is_only_a_review_flag():
    assert qa8_possible_missed_set("Mary carries the apple and milk", ["apple,milk"], 0)
    assert not qa8_possible_missed_set("Mary carries the apple and milk", ["apple,milk"], 1)
    assert not qa8_possible_missed_set("Mary carries the apple or milk", ["apple,milk"], 0)
    assert not qa8_possible_missed_set("Mary carries apple but not milk", ["apple,milk"], 0)
    assert not qa8_possible_missed_set("apple, milk, football", ["apple,milk"], 0)
    assert not qa8_possible_missed_set("nothing", ["nothing"], 0)


def test_count_synonyms_and_multiple_counts_are_review_flags_only():
    assert qa7_count_screen("There are 2 objects", ["two"], 0) == (True, False)
    assert qa7_count_screen("There are not 2 objects", ["two"], 0) == (False, False)
    assert qa7_count_screen("There are 2 or 3 objects", ["two"], 0) == (False, False)
    assert qa7_count_screen("There were 3 but now 2", ["two"], 1) == (False, True)
    assert qa7_count_screen("There are two objects", ["two"], 1) == (False, False)
