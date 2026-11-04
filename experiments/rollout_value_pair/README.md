# Qwen: jointly tune rollout length and Value update strength

The completed paired QA3 audit found that extending the student prefix from 8
to 32 tokens substantially increased the first-step SD gradient, while the
two-step cartridge answer stayed unchanged. A separate 8-token experiment
showed that a 0.001 Value learning rate moved the output more than 0.0001, but
did not improve the overall Qwen development accuracy. These observations
motivate testing their combination; neither observation proves it will help.

| Setting | Baseline | Candidate |
|---|---:|---:|
| Student rollout maximum | 8 | 32 |
| Value learning rate | 0.0001 | 0.001 |
| Base SD steps | 2 | 2 |
| Gain learning rate | 0.01 | 0.01 |

Both arms use the same loaded Qwen model and GPU, the same cached Decision
scores, 10% linked selection, original evidence-verified practice generation,
selected-context ICL Teacher, full-document Student, on-policy forward KL,
and original-question standalone generation. Each independently prefills the
two views, trains only its selected read parameters, materializes the selected
KV and releases full KV, selected source KV and training state before answering.
The runner checks that Teacher output tokens, verified practices and zero-update
outputs match; it does not claim the full Teacher probability tensors were
compared. Backbone weights and original document KV are frozen.

One one-GPU job evaluates 200 fixed development cases. Only if the candidate
beats the same-run whole, selected zero, ICL Teacher and original cartridge
controls will the hourly monitor submit a single 800-case source-disjoint
holdout job. The development set selects the configuration. Two holdout source
questions and aggregate results for other configurations have been examined
before, so it is not a fully blind holdout. Reference answers are used only
after generation for the common `answer_focused_lexical_v2` scorer. No paid
API calls are made. The single candidate changes two hyperparameters together;
its result will not isolate their individual contributions.

The development job is **32012882**, submitted for one GPU with a six-hour
allocation and requeue support. Frozen-runtime checks passed 82 tests, with
five GPU-specific cases skipped on the CPU; the allocation itself must run the
CUDA tests and real-model preflight before any dataset cases. The job uses
cached Decision scores and has no online API access.

If this submitted job starts or restarts after the scratch-cache configuration
was installed, its Triton and Torch extension caches are written under its own
scratch directory. An interpreter that was already running is not changed.

The GPU allocation runs real-model preflight checks before evaluation. The
report records the same-run four controls, paired new correct and new wrong
counts, and task/length breakdown. The goal is performance improvement, which
must be established by those results rather than inferred from gradient size.

[Prior paired audit](../../docs/analysis/paired-rollout-length-20261104/README.md)
· [Runner](runner.py) · [Report](report.py) · [Job](job.py) · [Monitor](monitor.py)
