# Claim–evidence review

| Claim | Evidence | Status |
|---|---|---|
| The saved query-specific questions share query terms more often. | Recomputed per-request word-set intersections, all 107 pairs. | Supported for this lexical metric and cohort. |
| Query-specific records more often co-present separated source regions. | Exact source-span collection; merged interval counts. | Supported as a construction diagnostic. |
| Improvements remain after removing original-query fallbacks. | Paired 85-request sensitivity table. | Supported for these two diagnostics only. |
| The dataset solves semantic dilution or fragmentation. | No semantic relevance or necessity test in this figure. | Not established; not claimed. |
| Contextual has better evidence completeness, correctness, speed, or accuracy. | None of those outcomes is measured here. | Not established; not claimed. |

## Five review questions

1. **Contribution:** What is new evidence here? A transparent description of the current saved records, not a new algorithm or proof of downstream benefit.
2. **Clarity:** Are definitions reproducible? The token rule, full stopword list, interval rule, denominators, and estimator are shipped with the records and code.
3. **Experimental strength:** Are the units independent and representative? No; 107 fixed requests share nine documents. The report makes descriptive cohort claims only.
4. **Completeness:** Are fallbacks and design-induced zeros hidden? No; all records remain in the main figure, with paired fallback sensitivity and an explicit explanation of Vanilla zeros.
5. **Design soundness:** Could these metrics improve without useful learning? Yes. Copying query terms or attaching irrelevant distant text can improve them. Semantic correctness, reasoning necessity, and SD effects require separate evidence.
