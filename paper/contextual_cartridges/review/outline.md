# Draft outline and evidence map

1. Setting: query-conditioned native KV memory with a frozen backbone.
2. Motivation: limited task-specific supervision of cross-source relations and diluted study budgets; interference from unconditional teacher imitation.
3. Technique 1: source selection, answer-first extraction, reverse question generation.
4. Shared objective: official sparse soft-target context distillation on aligned answer prefixes.
5. Technique 2: leave-one-out gradient nomination at three checkpoints, two-order deletion refits, survivor-loss guards, exact ordinary-SD fallback.
6. Completed evidence: matched synthetic material coverage, co-exposure, and footprint only.
7. Pending evidence: paired 2x2 BABILong accuracy and three-model, three-dataset cache curves.

| Claim | Evidence | Status |
|---|---|---|
| Fewer study contexts omit all query support | Figure 1 left; 1,000 paired requests | Observed source-coverage proxy |
| More exercise contexts co-expose all required facts | Figure 1 right; 352 multi-fact requests | Observed; not necessary multi-hop reasoning |
| Improved document-level support coverage | Figure 3 left; 994 paired requests | Observed; six missing pairs excluded |
| Smaller source footprint | Figure 3 right; 1,000 pairs | Observed characters, not timing |
| Faster generation under matched hardware | Requires matched end-to-end timing, including Decision cost | Not established here |
| Better final answer accuracy | Complete paired BABILong 2x2 | Pending |
| Selective SD corrects more answers than it breaks | Paired predictions under one frozen protocol | Pending |
| Better cache/performance tradeoff across backbones | Three models, three datasets, four capacities | Pending |

Review: the contribution is the query-conditioned construction and the forward-only selective-distillation protocol, not a new divergence theory. Every term is defined before use. Independent benchmark labels are excluded from selection. Selected refits use equal final update counts but higher total computation. Missing results remain null. No correct-to-wrong guarantee is claimed.

Questions before submission: Are the ablations complete? Are all capacity points evaluated on the same cases? Are fixes and regressions both reported? Is the full online cost measured? Are paper baselines matched by backbone and evaluation protocol? Has answer correctness and actual cross-source dependence been independently audited?
