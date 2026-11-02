# Implementation contract

This package extracts the method used by the submitted OpenAI-selection /
Selective-SD configuration. Its source lineage is recorded in
[`source_provenance.json`](source_provenance.json).

## The actual parameterization

For each layer and KV head, the fixed value basis contains the leading
eigenvectors of the selected native values' **uncentered second moment**:

$$U=\operatorname{eig}_{r}\left(\frac1{|S|}\sum_{u\in S}v_{u,0}^{\top}v_{u,0}\right).$$

Eigenvector signs are canonicalized. The basis is computed once, without answer
labels. Only the per-chunk coefficients $B_c$ and chunk score biases $g_c$ are
optimized. All coefficients and biases start at zero. Biases are clamped to
$[-6,6]$; value residuals are clipped separately at each selected position.

The allocation tensor includes entries for every source chunk for straightforward
indexing. Unselected entries are masked out of the forward computation and
constrained to zero. Value coefficient banks exist only for selected chunks.

For Qwen3-4B-Instruct-2507, the default uses 36 layers, 8 KV heads per layer,
head dimension 128, and rank 8. The package supports smaller full-attention
Qwen3 configurations for tests.

## Matching the deployed operator

Document K/V are written once. Every reader owns independent module containers
and suffix-cache objects, sharing only frozen backbone tensors.

At exactly zero adaptation, evaluation and final generation use the original
native model operator. Gradient evaluation uses the augmented attention path,
whose padding can introduce a finite-precision difference at zero. The checked
optimizer records this boundary and accepts proposals using the actual deployed
loss. At nonzero state, its gradient and no-gradient logits must match exactly.

Training and final generation both use causal incremental suffix execution.
Checkpoint recomputation reconstructs private branches; it cannot append into
another step's cache or expose the current target token. For long contexts,
layer checkpointing avoids retaining an augmented copy of every document layer
for backward.

## Source admission and update acceptance are separate

**Admission** happens once, before any update. The frozen support check collects
focused and full-context source views, performs candidate-blind source extraction,
and verifies the candidate against source quotes plus the original task. Its
final supported state receives the configured update budget; other states receive
zero. Quotes must occur in the original source after whitespace normalization.
The frozen LLM still performs the semantic assessment.

**Acceptance** happens within each optimizer iteration. It checks actual teacher
KL reduction, finite values, the output trust radius, and any activated query
prefix constraint. It may reject a proposed step and restore the previous state.
Accepting an optimization step is a statement about these measured objectives.

The inherited generic verifier, superseded by the task-aware verifier, is omitted.
Its verdict never entered the final verifier's prompt or the selected budget;
the source extraction and verified quotes are retained.

## Fixed target and joint update

The teacher is generated once from fresh selected-context KV. Its full-vocabulary
probabilities remain fixed. The loss is the mean of ordinary
$\mathrm{KL}(p_T\Vert p_S)$ at teacher answer positions, with the same preceding
answer tokens on both sides. There is no teacher/native probability mixture.

The submitted update uses Adam-preconditioned bias/value directions and a
two-direction output-Fisher model. Finite-difference checks assess the local
model; actual-loss backtracking checks each proposal. The Fisher's probability-
weighted centering is a covariance calculation inside the optimizer, not a
replacement or centering of the SD objective.

There is no cumulative whole-answer native-KL anchor. The scoped query-prefix
guard may constrain a confident, question-linked prefix after a shared qualifier.
The main defaults retain that behavior from the submitted implementation.

## Context and timing

Normal requests preserve the entire original document in student KV. Inputs
that exceed the configured context limit use the explicit overflow adapter in
`selection.py`. It retains API-ranked verbatim source in chronological order,
then records original and working hashes, lengths, and a source map. Comparisons
must use matching working inputs.

The independent package removes experiment dispatch, baseline inference,
reporting code, and redundant pre-update KL evaluations. It retains the teacher,
support prompts, read operator, KL target, and coupled update arithmetic.
The outer pipeline is expressed directly rather than through experiment-version
inheritance. Its timing is measured anew, rather than copied from an old run.

Validation details are in [`extraction_validation.json`](extraction_validation.json).
The numerical comparison uses synthetic CPU models. The extraction has not been
rebenchmarked as a new full-model GPU experiment.
