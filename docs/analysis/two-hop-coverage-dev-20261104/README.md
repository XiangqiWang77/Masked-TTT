# Two-hop selection: development coverage audit

This CPU-only exploration tests whether entity links can retain additional
annotated supporting facts without increasing the Decision-selected chunk
budget. It uses 200 Qwen development cases from 50 source questions, each at
four context lengths. All variants select 2,449 of 23,474 chunks in aggregate
(10.43%). The selector uses the question, document, chunk spans and cached
Decision scores; supporting-fact annotations and saved answer outcomes are
consulted only after selection.

| Selection | Changed cases | Complete support | Newly complete | Lost complete | Net change |
|---|---:|---:|---:|---:|---:|
| Existing entity-linked selection | — | 157 / 200 | — | — | — |
| Two-hop links | 54 | 154 / 200 | 0 | 3 | −3 |
| Two-hop links + score protection | 49 | 154 / 200 | 0 | 3 | −3 |

The score-protected variant was devised after the first audit and is
exploratory. Both variants fail the existing development coverage gate; neither
is adopted in the model experiments. Support coverage is a diagnostic, not an
answer-accuracy measurement. The 200 examples share source questions, and the
new selectors have not been evaluated on a disjoint holdout. No new model or
API calls were made.

The original entity-linked results and unprotected two-hop results match
between the [first snapshot](initial_summary.json) and the [combined
audit](summary.json) across every reported subgroup. The existing selection
remains unchanged for all queued and running experiments.
