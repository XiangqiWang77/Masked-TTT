# Same-budget question-guided evidence coverage

This CPU-only exploration expands the previously tested simple-location
question rule to two additional question forms: asking what a named person is
carrying, and asking about a giving event. The selector receives question text,
document text and cached Decision scores; it does not receive task IDs,
answers or support labels. For a matching question it reuses the existing
question-word overlap selection; otherwise it retains entity-linked selection.

Each policy selects exactly the same number of chunks per case: 2,449 of
23,474 across 200 development cases from 50 source questions. Supporting-fact
annotations are consulted only after selection.

| Candidate | Cases with changed selection | Complete support coverage | Newly complete / lost vs. entity-linked |
|---|---:|---:|---:|
| Entity-linked baseline | 0 | 157 / 200 | — |
| Location-question rule | 7 | 163 / 200 | 6 / 0 |
| Location + carrying | 18 | 165 / 200 | 8 / 0 |
| Location + carrying + giving | 37 | 166 / 200 | 9 / 0 |

Compared with the location rule, the final candidate covers three additional
cases completely; one is in the group where the saved full-context model was
correct and the saved Teacher was wrong. The saved Teacher had already
answered the other two correctly. The three cases come from three source
questions. This is an annotation and saved-output join, not an answer test of
the new selector. [Outcome audit](outcomes.json) · [script](../../../experiments/babilong_linked_selection/focused_question_outcomes.py).

The additional question patterns were
designed after reviewing these same development outcomes. They are exploratory
and have not been evaluated with a model or on a disjoint holdout; support
coverage is not answer accuracy.

The existing coverage screen is defined relative to entity-linked selection,
and both new candidates pass that screen. This does not establish a gain over
the location rule in model answers. Neither candidate has been adopted or
submitted; all current jobs retain their original selector. There were no new
model or online API calls.

[Aggregate data](summary.json) · [Selector](../../../experiments/babilong_linked_selection/focused_question_selector.py) ·
[Coverage audit](../../../experiments/babilong_linked_selection/focused_question_coverage.py) ·
[Previously completed location-rule answer comparison](../qwen-proper-where-dev-20261104/README.md).
