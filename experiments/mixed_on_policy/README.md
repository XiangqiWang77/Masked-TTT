# Mixed on-policy Student prefixes: unsubmitted candidate

Some completed BABILONG logs show that an eight-token greedy Student prefix can
end before the answer-bearing token. A separate 8/32-token comparison increased
the gradient on some questions without changing their final cartridge answers.
Those observations motivate testing another source of on-policy prefixes; they
do not establish an accuracy improvement.

This isolated candidate keeps the original method: the frozen Student reads
the full document, the frozen Teacher reads the same selected evidence as before,
and the Teacher scores the exact Student-generated prefix under full-vocabulary
forward KL. Only selected chunk gains and low-rank Value coefficients are
updated; the trained state is materialized onto an independent selected-only KV
cartridge, and the full/source caches and optimizer state are released before
the original question is answered. No evaluation labels or online API are used.

The original first update remains greedy. The second update samples from the
current Student's full-vocabulary distribution at temperature 1; further calls
alternate greedy and sampled prefixes. Sampling uses a private generator with
the request's fixed seed and does not alter global RNG. The first update thus
matches the baseline, and the second explores another on-policy trajectory.
The context manager patches a process-local function and is restricted to
single-request experiments; it restores the original function on failure.

The same-device pair validates identical selection, Teacher output tokens,
verified source practices and zero-update output; it does not claim that full
Teacher probability tensors were compared. Qwen and GPT-OSS miniature-model
tests check the first update, causal prefixes, gradient path, reproducibility,
frozen state and physical cache release. This candidate is **not submitted or
enabled in the current benchmark jobs**. It is a possible subsequent controlled
test if the already submitted configurations do not achieve the target.

[Implementation](rollouts.py) · [Tests](../../tests/test_mixed_on_policy.py) ·
[Prior paired rollout audit](../../docs/analysis/paired-rollout-length-20261104/README.md)
