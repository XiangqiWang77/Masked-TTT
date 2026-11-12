# Contextual Cartridges: ED-JSD Selective Self-Distillation

Contextual Cartridges combines the fixed query-specific synthetic dataset with
ED-JSD selective SD. The BABILong controls are query-agnostic/specific data
crossed with ordinary SD/ED-JSD SD. The cache study runs Contextual Cartridges
only on LongHealth, MTOB (KE), and QASPER, at 512/1024/2048/4096 native KV slots.

For a budget of T original SD updates, perform h=floor(T/2) warmup updates on
all trajectories. Trajectories whose recorded deterministic Source-Support
Gate decision is a rejection (empty, truncated, or explicitly refused Teacher
answers, recorded at Teacher generation) are excluded from the candidate pool
before scoring; they still participate in warmup and in the exact full-data
fallback. At the resulting fixed KV state, run one forward-only pass
per remaining candidate, using exactly the same answer prefixes as the Teacher.
At each answer position, retain every stored Teacher probability and the
corresponding full-vocabulary-normalized Student probability. Append the
omitted probability mass as one rest bucket to each distribution.

With base-2 entropy and M=(T+S)/2, score a trajectory by the token mean of:

`[H(M) - (H(T)+H(S))/2] * [H(S)-H(T)] / [H(S)+H(T)+1e-12]`.

Admit trajectories with score strictly greater than zero. Continue the
remaining T-h updates on admitted trajectories, using the same Adam state.
If none are admitted, continue the original data sampler and optimizer
unchanged. If all are admitted, also keep the original sampler unchanged.
These two branches reproduce ordinary SD under deterministic forwards.
There is no second fit, TracIn, cross-trajectory gradient comparison,
heldout reference loss, benchmark label, hard-label answer loss, or RL update.
The training objective stays the upstream sparse soft-target SD objective;
JSD is a selection statistic, not the optimization loss.

The gate's entropy and JSD are **coarsened-distribution statistics**, not exact
full-vocabulary statistics. Aggregating the tail can hide disagreement within
the tail, and can change the apparent entropy ordering. Missing Teacher
probabilities are not assumed zero and retained top-k mass is not renormalized
to one. Stable `0 log 0 = 0` handling keeps the bucket statistic finite.

Confidence is not correctness. A confidently wrong Teacher can pass this gate;
a confidently wrong Student can cause correct Teacher information to be
rejected. JSD is not the only numerically usable divergence; forward KL can
also be computed stably using log-softmax. No claim of guaranteed accuracy
improvement, minority-fact protection, or universal non-regression is made.
The empty-gate fallback is an exact continuation only for that branch.

The six-hour allocation records selection time/forward counts separately
from the T backward updates. It does not imply that 1,000 four-arm requests
plus all cache cells will finish within six hours. Completed results retain
their own protocol and only identical ordinary checkpoints may be reused.
