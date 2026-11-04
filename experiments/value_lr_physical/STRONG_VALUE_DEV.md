# GPT-OSS original SD: Value learning rate 0.003

In the completed 200-case paired development study, increasing the Value rate
from 0.0001 to 0.001 changed more GPT-OSS answers and moved its accuracy from
52.50% to 54.00%; the same 0.001 setting did not improve Qwen. The input-level
and token-level checks are documented in the [Value-strength audit](../../docs/analysis/value-strength-audit-20261104/README.md).

This fixed GPT-OSS candidate tests rate **0.003** while preserving the original
method: cached Decision score top 10%, independent selected Teacher and full
Student caches, an eight-token on-policy Student prefix, full-vocabulary
Teacher-to-Student KL, two base steps plus verified practice steps, gain rate
0.01, and updates restricted to the selected KV reading state. The final answer
comes from a newly materialized independent selected cache after the full cache,
source selected cache and training adapter are released. There is no extra
full-to-selected loss and no evaluation answer or network call during inference.

Development job **31989997** uses one GPU for the same 200 cases. The reporter
compares it with the earlier 0.001 outputs only when selected chunks, Teacher
tokens, verified exercises, initial Student prefix and zero-update output agree.
Teacher, full-context and zero-update controls are also reported on every valid
candidate case with one task-aware loose lexical metric. Teacher retains its
original generic response-format instruction; zero and Cartridge use the same
original question format. The candidate is an experiment, not a guaranteed gain.

The frozen runtime passed **74 CPU tests** and performs a loaded-model GPU
preflight before evaluation. If all 200 answers are valid and the candidate
exceeds its same-run Teacher, full model and selected zero-update baseline, the hourly
monitor submits one single-GPU **800-case source-disjoint holdout**. The combined
1,000-case report will distinguish the 200 development cases used for selection
from the 800 holdout cases; two holdout source questions were inspected before
this experiment. Otherwise the monitor records the outcome and does not promote.
The comparison with 0.001 remains a separately reported matched subset, since
different GPU runs may generate different self-study exercises.

[Runner](strong_dev_runner.py) · [Report](strong_dev_report.py) ·
[Preparation](strong_dev_prepare.py) · [Monitor](strong_dev_monitor.py)
