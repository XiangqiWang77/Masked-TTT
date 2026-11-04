# Archived and physical cartridge runtime compatibility

The 200-case Value-rate development sweep used an older paired-output runtime. The new 800-case holdout uses the same selected-teacher/full-student on-policy KL but physically releases the source and training state before decoding. This check tests whether that runtime change preserves the training trajectory and answer in a controlled setting.

Four synthetic CPU experiments compare Qwen3 and grouped-query GPT-OSS miniature architectures at Value learning rates `0.0001` and `0.001`. Every pair produced the same Teacher token sequence, Student on-policy prefixes and final selected-cache answer. The maximum absolute difference among the logged KL and gain/Value gradient norms was **0** in these CPU checks. The new path additionally verified that no full-document KV, duplicate source KV or training-adapter storage remained at standalone decoding.

The synthetic cases do not use evaluation answers, saved Decision scores, an online API or real model checkpoints. The submitted GPU job separately runs its loaded-model preflight. This comparison checks the runtime transition; it is not a claim of a downstream accuracy improvement.

The frozen GPT physical holdout and the 200-case step-development runtime were also checked with miniature models matching Qwen3-4B's 4:1 and GPT-OSS-20B's 8:1 query/KV head ratios. Fourteen CPU checks for each frozen runtime passed: chunked prefill reproduces the unmodified Transformers model; the differentiable zero-correction path reproduces its native suffix read; and gradients remain finite with base weights frozen. This is an architectural test, not a substitute for the actual checkpoints' GPU preflight or the BABILONG results.

[Results](results.json) · [Reproduction script](../../../experiments/value_lr_physical/archive_equivalence.py)
