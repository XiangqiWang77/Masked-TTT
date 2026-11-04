# Backbone position settings

The public loader preserves the position configuration in the checkpoint and
uses `min(requested_context, max_position_embeddings)` as its working limit.
Previously the Qwen path overwrote both the position limit and `rope_scaling`.
This could silently discard an explicitly configured Qwen RoPE scaling scheme.
The GPT-OSS loading path retains its existing YaRN settings.

The native GPT-OSS checkpoint declares MXFP4 quantization, which is still
dequantized for differentiation. An already unquantized GPT-OSS checkpoint
now loads directly; forcing the MXFP4 loader on it previously looked for
quantized expert blocks that were not present. The local GPT-OSS-20B config
used by the submitted jobs declares `quant_method: mxfp4`, YaRN RoPE, and a
131,072-position limit, so its loading branch remains the same.

The Qwen3-4B-Instruct-2507 checkpoint used in the current BABILONG runs has a
native limit of 262,144 positions, `rope_scaling: null`, and `rope_theta:
5000000`. With a requested limit of 131,072, a direct comparison of the old
and new RoPE configuration produced exactly the same inverse frequencies,
cosines and sines at positions 0, 8,192, 65,535 and 131,071. This comparison
loads only the local model configuration; it makes no model or API calls.

[Tests](../../../tests/test_backbone_context.py) also verify preservation of
scaled Qwen RoPE and GPT-OSS YaRN, native context limits, and matching logits
for the current Qwen configuration on a small model. An additional
[offline checkpoint test](../../../tests/test_local_checkpoint_loading.py)
loads and adapts small saved Qwen and unquantized GPT-OSS models end to end.
Already submitted jobs
retain their frozen runtime. The synthetic and configuration checks do not
replace the pending real-model GPU preflight or evaluate answer quality.
