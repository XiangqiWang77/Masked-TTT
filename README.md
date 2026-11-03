# Query Cartridges: Learning to Read Long Contexts with Masked Test-Time Training

Query Cartridges adapts a frozen language model to one long-context question.
Question-conditioned Decision scores select a small number of document chunks.
A frozen, selected-context ICL teacher supplies on-policy next-token
distributions and proposes short source-grounded practice questions. The
full-context student jointly adapts two request-local reading corrections:
chunk-level attention allocation and a low-rank value-read adjustment. Model
weights and the original document KV remain immutable.

This branch preserves the archived implementation used in the OLMo BABILONG
run. The recorded profile is available at
[examples/olmo_babilong_icl_sd.json](examples/olmo_babilong_icl_sd.json).
The [Chinese method description](docs/OLMO_ICL_SELF_DISTILLATION.zh-CN.md)
explains the two cache paths, the paired results, and their scope.

The trained state supports two explicitly different answer paths:

* **Full KV + Query Cartridge:** answer from the working-document cache with
  the trained corrections. A frozen full-context judge compares the native and
  adapted answers in both orders; an inconclusive comparison retains the native
  answer.
* **Standalone Query Cartridge:** answer without reading the full-document KV.
  It contains an independently prefetched selected-chunk KV together with the
  trained corrections, materialized for decoding. Its answer is evaluated
  separately from the full-context path.

The standalone cartridge is a real selected-KV cache, not just a small set of
parameters. It is constructed after full-context adaptation; consequently its
smaller decoding state does not by itself establish a lower training-time peak
memory requirement. Unlike the reusable virtual caches in
[Cartridges](https://arxiv.org/abs/2506.06266), these states are constructed for
one request and discarded afterwards.

## Procedure

1. Load the saved Decision API scores for the document and question. Select
   the top 10% of source chunks in their original order.
2. Prefill the complete working document once. Separately prefill the selected
   text for the ICL teacher; obtain its answer for the baseline comparison.
3. Ask the teacher for up to two short factual practice questions, answers and
   exact source quotations. Keep only proposals whose quotation occurs in a
   selected chunk and whose answer agrees with a separate frozen-model answer
   based on that quotation alone. This is a consistency check, not a guarantee
   of logical entailment; evaluation references are never used here.
4. Generate short current-student prefixes for the original and verified
   practice questions. Score each identical prefix with the focused teacher and
   minimize full-vocabulary forward KL on the student path. All exercises share
   the same selected-position attention and value-read state.
5. Discard training suffixes and independently generate the original answer.
   The full-context judge sees both the adapted and native answers, reads the
   working document and can retain the native answer. Also generate an
   independent answer with the standalone selected-KV cartridge.

At a selected document location $u$ in chunk $c$, the adapted read is

$$z_{t,u}=q_t^\top k_{u,0}/\sqrt{d_h}+m_u g_c,\qquad
o_t=\sum_u \operatorname{softmax}(z_t)_u\left(v_{u,0}+
m_u\operatorname{clip}_{0.15\lVert v_{u,0}\rVert}((v_{u,0}U)B_c)\right).$$

The mask $m_u$ limits trainable changes to selected document positions; the
fixed basis $U$ is computed from their cached values. Full-attention layers
are adapted and sliding-attention layers retain their native behavior.

## Memory accounting

Each result records the backing storage of the full document KV, the
request-local adapter, optimizer state, and the materialized decode cache.
Standalone cartridge size includes both selected KV and its adapter. Shared
tensor views are counted once. A GPU peak records the full paired experiment
and includes model weights and training; it is not presented as a separately
measured standalone peak. Source and question/output suffix KV are identified
separately so that the compared cache scope is clear.

## Run

Install with `pip install -e '.[test]'` and a compatible FlashAttention build.
The saved score directory contains `case_to_score_file.json` and `scores/`.

```bash
masked-ttt --model /path/to/model --request request.json \
  --score-json-dir /path/to/shared_scores --case-id YOUR_CASE_ID \
  --config-json examples/olmo_babilong_icl_sd.json \
  --local-files-only --output result.json
```

The full answer is `prediction`; the independent selected-KV result is
`standalone_cartridge.prediction`. The `self_study`, `self_judge`, `training`,
and `memory` fields record the actual request path. If a document exceeds a
backbone's native window, the cached selector constructs a source-prioritized
working document and records its source mapping.

## Verify

Run `pytest -q`. Evaluation compares paired answers with the same task-aware
metric, reports gains and losses relative to each baseline, and reports
document KV and cartridge storage separately. Earlier OPD-only and
teacher-prefix runs are different configurations, not results of this method.
