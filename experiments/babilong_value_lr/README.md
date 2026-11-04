# BABILONG Value learning-rate calibration

This experiment keeps the Query Cartridge method fixed and tests whether its
Value update needs a different learning rate on GPT-OSS-20B and Qwen3-4B.
The document Student, selected ICL Teacher, on-policy KL, trainable attention
bias and Value coefficients, selected-only materialization, and full-KV
release are unchanged. Decision scores are reused from the existing cache.

The three Value learning rates are `0.0001`, `0.0003`, and `0.001`; the
attention-bias learning rate stays `0.01`, and the fixed low-rank basis is
constructed from the selected positions in the full-document KV. There is no
additional distillation loss or training directly on the selected cache.

[`split.json`](split.json) records the 1,000 public case identifiers and the
predeclared split. The development set has 200 cases from 50 source questions;
the 800-case holdout has 200 disjoint source questions. Each source's four
length variants stay together. Selection is stratified across the ten BABILONG
question types and four lengths. The earlier fixed 40-case development probe
is included in the 200-case development set, never in the holdout.

Each development example is evaluated with all three rates using the same
model, cached chunk selection, Teacher, selected zero-update control, original
question format, and GPU allocation. Incomplete or mismatched pairs do not
enter the report. Once all 200 paired examples are available, a rate qualifies
only if its accuracy exceeds KV Whole, selected zero-update, and ICL Teacher.
Choose the highest development accuracy among qualifying rates; break ties by
fewer new errors relative to selected zero-update, then by the smaller rate.
If no rate qualifies, no rate is promoted.

For each qualifying model, the chosen rate and original rate are then evaluated
on the 800 source-disjoint holdout examples without further tuning. Report all
new correct and new wrong predictions and the cache storage ratio. The intended
holdout target is at least **+3 percentage points versus ICL Teacher**, while
also exceeding KV Whole and selected zero-update. The 1,000-case combined table
must retain separate 200-case development and 800-case holdout rows. All
answers and support annotations are used only after generation for scoring.

The experiment has been submitted; these are test criteria, not reported results.
