# Same-budget two-hop evidence candidate

The completed linked-selection development analysis showed that recovering
missing support facts often changed answers. This experimental selector tests
whether retaining an additional entity-linked passage can improve coverage
within the same approximately 10% Decision-score budget.

It begins with the existing entity-linked selection and never removes its
highest-score anchors. Names from question-related anchor sentences identify
first-hop entities; a sentence in an already selected block that connects one
of those entities to another name supplies a second-hop bridge. The candidate
may replace at most half the reserved linked slots with a block mentioning the
second-hop entity. It preserves both every selected bridge source and any
previously added destination. If there is no room to retain the linking block,
the original selection remains unchanged. It receives only document text,
question, source chunk spans and cached Decision probabilities.

The first exploratory audit changed 54 of 200 selections and reduced complete
support coverage from 157 to 154, with no newly complete cases. It was rejected
for model testing. A subsequent exploratory variant additionally allows a
replacement only when its cached Decision score exceeds that of the block it
would remove. This score protection was designed after the first audit; its
development result must not be described as preregistered or held out. It
changed 49 selections and also reduced complete coverage from 157 to 154.
Both variants failed the coverage gate and were not submitted for model testing.

The 200-case development audit computes selection first and checks annotated
support only afterward. It compares the same chunk budget and uses the existing
coverage threshold; the new heuristic was designed after previous development
analyses, so this is exploratory. Support coverage is not answer accuracy.
There are no new model or online API calls. No GPU job or production selector is
changed by this candidate.

[Selector](two_hop_selector.py) · [Coverage audit](two_hop_coverage_dev.py) ·
[Tests](../../tests/test_two_hop_selector.py) ·
[Aggregate results](../../docs/analysis/two-hop-coverage-dev-20261104/README.md)
