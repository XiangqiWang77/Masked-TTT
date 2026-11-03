# Implementation notes

`FrozenJSONSelector` maps shared Decision API source spans to each model's
native tokenization and records any working-document compaction. The GPU
evaluation path has no network fallback.

`teacher.py` prefills selected source tokens independently and scores the
student's current causal prefixes. `self_study.py` and
`self_study_runtime.py` create up to two source-quoted factual exercises,
require the answer to occur in the quote, and compare it with an independently
generated quote-only answer. Practice paths are admitted only if their
document token prefixes exactly match the caches being reused.

`teacher_judge.py` and `teacher_judge_runtime.py` optionally check a focused
Teacher's proposed answer before the main-question update. The Qwen V5 profile
uses a source-grounded check and a separate full-context comparison against
the native answer. When configured, verified practice questions remain
eligible even if the main Teacher signal is rejected. These judgments use no
benchmark labels and do not establish logical correctness by themselves.

`pipeline.py` constructs one request-local `KVReadMemory` and trains the main
and practice question paths jointly using full-vocabulary forward KL. The
default profile has two main steps plus one step per verified exercise, up to
eight steps for an identified multiple-choice answer. Student rollouts are
refreshed on every step. The original model weights and document cache are
read-only.

`output_guard_runtime.py` compares the adapted and native answers using the
full working document and both candidate orderings. It reads the frozen
model's A/B/U distribution and accepts the adaptation only when both orders
favor it with conditional confidence at least 0.65 and label mass at least
0.20. These are configuration defaults, not ground-truth checks.

`standalone.py` maps the original selected chunk IDs onto the independently
prefilled selected-document KV. It transfers each selected chunk's learned
attention and value coefficients, materializes the selected KV for decoding,
and evaluates an independent answer without accessing the full-document KV.
Because the selected-only model prefix is recomputed, it may differ from
selected positions of the full-context cache.

`memory_accounting.py` counts distinct tensor backing storage. Full document
KV, adapter state, optimizer state, full decode materialization, selected KV
and standalone materialization are reported separately. The GPU peak is for
the complete paired run, including model and training; a separate standalone
peak is not inferred from it.

The recorded Qwen3-4B BABILONG V5 profile is provided as
`examples/qwen_babilong_v5_config.json` and can be loaded with the CLI's
`--config-json` option. It sets a 30% selected fraction, four base updates,
up to 16 answer updates and 32 Student rollout tokens. The `babilong-v1` Git
tag preserves the separate earlier base implementation and its defaults.

The implementation supports Qwen3, OLMo3 and GPT-OSS. The current Qwen3 model
adapts its 36 full-attention layers; hybrid models adapt only full-attention
layers and preserve the native sliding-window behavior. Rank defaults to eight,
value residuals are bounded by 15% of the original value norm, chunk biases
are clamped to [-6, 6], and initial corrections are zero.
