# Same-method, paired selected-evidence tokenization

The completed source-only audit found that both Qwen3-4B and GPT-OSS-20B can build a canonical selected-fragment token plan for all 200 fixed development cases. For each model, 197 of those cases tokenize differently from the original method's selected-token splice. Both plans decode to the same evidence text. This is a candidate implementation detail, not evidence of an accuracy improvement.

The comparison runs both plans sequentially on the same loaded model and GPU. Each uses the same cached Decision selection, frozen full-document Student, selected-only Teacher, on-policy KL, chunk gains, Value update, and original-question final answer. Each separately prefills and materializes its selected cache, then releases the full document cache and training state. The selected document fragment is retokenized; the original prompt prefix and question suffix token IDs remain fixed. This is not retokenization of the entire chat prompt.

Because the evidence token IDs change, Teacher predictions, verified practices and zero-update answers are allowed to differ. The report measures all of them separately, as well as the trained cartridges. References enter only the uniform `answer_focused_lexical_v2` post-generation scoring. The 200 cases use 50 source questions at four context lengths.

One GPU job covers both models; GPT-OSS is routed to a high-memory GPU when needed. The job performs real-model synthetic preflight for both paths. Only a model whose aggregate accuracy exceeds its same-run whole, canonical zero, canonical Teacher and original-SD controls on the complete 200-case development set is eligible for its source-disjoint 800-case holdout. Hourly monitoring avoids repeated manual checks. No online Decision API calls are made.

Development job **32010104** was submitted on 2026-11-04. Before submission, the frozen runtime passed **107 CPU tests**, and a fresh source-only audit built the candidate plan for 200/200 cases on each model. The loaded-checkpoint GPU preflight and answer-quality comparison are pending; there is no measured performance result for this variant yet.

[Comparison](canonical_pair.py) · [Runner](canonical_runner.py) · [Scoring](canonical_report.py) · [Submission](canonical_job.py) · [Monitor](canonical_monitor.py) · [Source-only feasibility](../../docs/analysis/evidence-tokenization-20261104/README.md)
