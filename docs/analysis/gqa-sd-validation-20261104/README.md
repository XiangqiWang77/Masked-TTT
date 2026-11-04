# Grouped-query self-distillation check

The current Qwen3-4B and GPT-OSS-20B checkpoints use grouped-query attention.
The [causal distillation tests](../../../tests/test_batched_sd.py) include
small Qwen and GPT-OSS models with the same query-to-KV grouping ratios,
respectively 4:1 and 8:1. The GPT-OSS test includes its learned attention sinks
and alternating local and full attention layers.

For short and 32-token student continuations, the tests compare batched
training with incremental token-by-token execution. They check causal logits,
joint gain and value gradients, immutable document KV and frozen model weights.
Changing later continuation tokens must not affect earlier predictions. The
selected Teacher is also compared with incremental evaluation on the same
Student prefixes.

All 31 tests passed on CPU with synthetic models. They check the computation,
not BABILONG accuracy; the real-model CUDA preflight remains scheduled in the
existing GPU jobs.
