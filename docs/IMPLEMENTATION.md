# Implementation notes

## Request path

1. `FrozenJSONSelector` binds a case ID, reads its precomputed score file,
   and maps the original character spans into the current model's tokenization.
   It has no API fallback. `OpenAIChunkSelector` is a separate online path.
2. `teacher.py` rebuilds a fresh selected-only document input. The teacher
   cache is reused; each update scores the current student-generated prefix.
3. `pipeline.py` stores the native working-document cache, constructs
   request-local `KVReadMemory`, trains selected reads and generates a
   fresh final answer.
4. `batched_sd.py` evaluates the whole question plus the student prefix in a
   causal batch. Layer checkpointing bounds activation storage during backward.
5. `memory.py` stores selected-chunk score biases and low-rank value
   coefficients. The base cache and model parameters remain unchanged.

The selected value basis is computed separately per adapted layer and KV head.
Its rank is eight by default. Each selected chunk has its own learned value
coefficient and attention bias. The value residual is clipped to 15% of the
original value norm and chunk biases are clamped to $[-6,6]$.

## Attention

After RoPE, the attention implementation appends an extra query component of
one and a matching key component proportional to the selected chunk bias.
The original attention scale is retained, giving exactly an additive bias in
the score. Extra value dimensions are zero padded and removed from the output.
The original cached tensors are not modified.

The current Qwen3 model adapts all 36 full-attention layers. OLMo3 and GPT-OSS
can mix full and sliding attention; the adapter targets their full-attention
layers and leaves sliding layers native. The full working document is always
read by the student when it fits in the backbone's context window.

## Profile

| Setting | Value |
|---|---|
| Source selection | top 10% of cached question-conditioned scores |
| Teacher | one selected-context generation, maximum 32 tokens |
| Teacher objective | full-vocabulary forward KL on current student prefixes |
| Student rollout | greedy, refreshed each step, at most 8 tokens |
| Base updates | 2 joint Adam steps |
| Choice-answer adaptation | up to 8 steps; stop when KL ≤ 0.5 |
| Learning rates | 0.01 for score bias; 0.0001 for value coefficients |
| Sink positions | first 4 prefix-cache positions excluded |
| Backbone | frozen |
| Training suffix | discarded before final generation |

For non-choice questions, all student-prefix positions are used and only the
base update budget applies. A choice position is used only when the question
format and Student rollout make it identifiable. This profile does not invoke a
separate teacher judge or perform NTP.

## Timing

`timing.ttft_seconds` starts at request handling and ends at the first
student token. It includes local JSON loading, working-context preparation,
native document prefill, teacher generation and SD updates. It excludes model
loading, scheduler queueing and the previously completed shared Decision API
preprocessing. The precomputed scoring time and estimated cost are separate
fields, so cached online and cold request costs can be reported separately.

`timing.icl_teacher_first_token_seconds_segmented` adds measured request setup
to the Teacher's prompt construction, selected-context prefill, and first-token
time; it excludes the intervening Student document prefill. Earlier runs did
not include Teacher prompt construction in their core timing, so those estimates
must be kept separate.

Earlier teacher-prefix jobs used a different runtime and are not OPD results.
This repository contains the method implementation rather than cluster
orchestration or evaluation labels.
